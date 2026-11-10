"""Language identification (LID) for user posts - interview lab module.

Question: "Design a language identification classifier for user posts
(100+ languages, short noisy text, code-mixing)."

What is real here
  * a small hand-written PARALLEL corpus (same 30 sentences in 12 languages, so topic
    cannot be used as a shortcut) + frequent-word lexicons + slangy "social" posts
  * synthetic training text made by sampling word windows / shuffling phrases
  * stage 1: Unicode-script rule; stage 2: char n-gram model restricted to the
    languages that script allows (a cascade)
  * three stage-2 models: TF-IDF(1-4 char) + Multinomial NB, TF-IDF + logistic
    regression, and a fastText-style hashed n-gram embedding bag written in numpy
  * abstention ("unknown") by confidence threshold, calibration (ECE), confusion matrix,
    accuracy-vs-length, class imbalance, and three classic pitfalls (leakage, domain
    shift, URL/hashtag noise) each with a wrong-vs-right toggle.

Everything is deterministic (fixed seeds) and built lazily on the first request.
"""
from __future__ import annotations

import re
import threading
import time
import unicodedata
import zlib
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import sparse  # hard dependency of scikit-learn; used only for the n-gram bag matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

META = {
    "slug": "language-classifier",
    "title": "Language Classifier",
    "tagline": "Script rule -> char n-gram cascade for short, noisy, code-mixed posts; abstention, confusions, pitfalls.",
    "icon": "\U0001F310",
    "order": 6,
}
router = APIRouter()

# =====================================================================================
# 1. DATA - hand-written parallel corpus (interview concept: where do labels come from,
#    and why a parallel / topic-balanced corpus stops the model learning topic != language)
# =====================================================================================
LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
    "it": "Italian", "hi": "Hindi (Devanagari)", "hi-Latn": "Hinglish (romanized Hindi)",
    "ru": "Russian", "uk": "Ukrainian", "ar": "Arabic", "ja": "Japanese",
}
LANGS = list(LANG_NAMES)
L_IDX = {l: i for i, l in enumerate(LANGS)}
NO_SPACE = {"ja"}  # languages written without spaces -> character windows instead of word windows

FORMAL = {
"en": """The government announced a new economic policy on Monday.
Scientists have discovered a new species of bird in the forest.
The meeting will take place at the university next week.
She has been working at the hospital for many years.
The weather forecast predicts heavy rain throughout the weekend.
Children should read books every day before going to sleep.
The company reported strong growth during the third quarter.
We would like to thank everyone who helped with this project.
The museum is open from nine in the morning until five in the evening.
He bought a small house near the river with his brother.
Public transport in the city has improved considerably.
Please remember to bring your passport and your ticket.
The president will speak to the nation tonight about the crisis.
Many people believe that education is the key to a better future.
The train to the airport leaves every thirty minutes.
Our team won the championship after a very difficult season.
The restaurant serves fresh fish and vegetables from the local market.
I think this is the best book that I have ever read.
The students are preparing for their final exams in the library.
There was a lot of traffic on the highway this morning.
Where can I find a good doctor in this neighbourhood?
The new law will protect the rights of workers and their families.
My grandmother told wonderful stories about her childhood in the village.
The price of oil has increased again this month.
Would you like a cup of coffee or some tea with milk?
Researchers are studying how the climate is changing around the world.
The film was shown for the first time at the festival yesterday.
They decided to travel through the mountains during the summer holidays.
What time does the shop close on Saturdays?
The children were playing in the garden when it started to snow.""",
"es": """El gobierno anunció una nueva política económica el lunes.
Los científicos han descubierto una nueva especie de pájaro en el bosque.
La reunión tendrá lugar en la universidad la próxima semana.
Ella trabaja en el hospital desde hace muchos años.
El pronóstico del tiempo anuncia fuertes lluvias durante todo el fin de semana.
Los niños deberían leer libros todos los días antes de dormir.
La empresa registró un fuerte crecimiento durante el tercer trimestre.
Queremos agradecer a todos los que ayudaron con este proyecto.
El museo está abierto desde las nueve de la mañana hasta las cinco de la tarde.
Compró una casa pequeña cerca del río con su hermano.
El transporte público de la ciudad ha mejorado mucho.
Por favor, recuerde traer su pasaporte y su billete.
El presidente hablará esta noche a la nación sobre la crisis.
Mucha gente cree que la educación es la clave para un futuro mejor.
El tren al aeropuerto sale cada treinta minutos.
Nuestro equipo ganó el campeonato después de una temporada muy difícil.
El restaurante sirve pescado fresco y verduras del mercado local.
Creo que este es el mejor libro que he leído en mi vida.
Los estudiantes se preparan para sus exámenes finales en la biblioteca.
Había mucho tráfico en la autopista esta mañana.
¿Dónde puedo encontrar un buen médico en este barrio?
La nueva ley protegerá los derechos de los trabajadores y sus familias.
Mi abuela contaba historias maravillosas sobre su infancia en el pueblo.
El precio del petróleo ha subido otra vez este mes.
¿Quieres una taza de café o un té con leche?
Los investigadores estudian cómo está cambiando el clima en todo el mundo.
La película se proyectó por primera vez ayer en el festival.
Decidieron viajar por las montañas durante las vacaciones de verano.
¿A qué hora cierra la tienda los sábados?
Los niños jugaban en el jardín cuando empezó a nevar.""",
"fr": """Le gouvernement a annoncé une nouvelle politique économique lundi.
Les scientifiques ont découvert une nouvelle espèce d'oiseau dans la forêt.
La réunion aura lieu à l'université la semaine prochaine.
Elle travaille à l'hôpital depuis de nombreuses années.
La météo prévoit de fortes pluies pendant tout le week-end.
Les enfants devraient lire des livres chaque jour avant de dormir.
L'entreprise a enregistré une forte croissance au cours du troisième trimestre.
Nous voulons remercier tous ceux qui ont aidé à ce projet.
Le musée est ouvert de neuf heures du matin à cinq heures du soir.
Il a acheté une petite maison près de la rivière avec son frère.
Les transports publics de la ville se sont beaucoup améliorés.
N'oubliez pas d'apporter votre passeport et votre billet, s'il vous plaît.
Le président parlera ce soir à la nation au sujet de la crise.
Beaucoup de gens pensent que l'éducation est la clé d'un avenir meilleur.
Le train pour l'aéroport part toutes les trente minutes.
Notre équipe a gagné le championnat après une saison très difficile.
Le restaurant sert du poisson frais et des légumes du marché local.
Je crois que c'est le meilleur livre que j'aie jamais lu.
Les étudiants préparent leurs examens finaux à la bibliothèque.
Il y avait beaucoup de circulation sur l'autoroute ce matin.
Où puis-je trouver un bon médecin dans ce quartier ?
La nouvelle loi protégera les droits des travailleurs et de leurs familles.
Ma grand-mère racontait de merveilleuses histoires sur son enfance au village.
Le prix du pétrole a encore augmenté ce mois-ci.
Voulez-vous une tasse de café ou un thé au lait ?
Les chercheurs étudient comment le climat change dans le monde entier.
Le film a été projeté pour la première fois hier au festival.
Ils ont décidé de voyager à travers les montagnes pendant les vacances d'été.
À quelle heure le magasin ferme-t-il le samedi ?
Les enfants jouaient dans le jardin quand il a commencé à neiger.""",
"de": """Die Regierung hat am Montag eine neue Wirtschaftspolitik angekündigt.
Wissenschaftler haben im Wald eine neue Vogelart entdeckt.
Das Treffen findet nächste Woche an der Universität statt.
Sie arbeitet seit vielen Jahren im Krankenhaus.
Der Wetterbericht sagt für das ganze Wochenende starken Regen voraus.
Kinder sollten jeden Tag vor dem Schlafen Bücher lesen.
Das Unternehmen verzeichnete im dritten Quartal ein starkes Wachstum.
Wir möchten allen danken, die bei diesem Projekt geholfen haben.
Das Museum ist von neun Uhr morgens bis fünf Uhr abends geöffnet.
Er hat mit seinem Bruder ein kleines Haus in der Nähe des Flusses gekauft.
Der öffentliche Verkehr in der Stadt hat sich deutlich verbessert.
Bitte denken Sie daran, Ihren Reisepass und Ihre Fahrkarte mitzubringen.
Der Präsident wird heute Abend zur Nation über die Krise sprechen.
Viele Menschen glauben, dass Bildung der Schlüssel zu einer besseren Zukunft ist.
Der Zug zum Flughafen fährt alle dreißig Minuten.
Unsere Mannschaft hat nach einer sehr schwierigen Saison die Meisterschaft gewonnen.
Das Restaurant serviert frischen Fisch und Gemüse vom örtlichen Markt.
Ich glaube, das ist das beste Buch, das ich je gelesen habe.
Die Studenten bereiten sich in der Bibliothek auf ihre Abschlussprüfungen vor.
Heute Morgen gab es viel Verkehr auf der Autobahn.
Wo kann ich in diesem Viertel einen guten Arzt finden?
Das neue Gesetz wird die Rechte der Arbeiter und ihrer Familien schützen.
Meine Großmutter erzählte wunderbare Geschichten über ihre Kindheit im Dorf.
Der Ölpreis ist in diesem Monat wieder gestiegen.
Möchtest du eine Tasse Kaffee oder einen Tee mit Milch?
Forscher untersuchen, wie sich das Klima auf der ganzen Welt verändert.
Der Film wurde gestern zum ersten Mal auf dem Festival gezeigt.
Sie beschlossen, in den Sommerferien durch die Berge zu reisen.
Um wie viel Uhr schließt das Geschäft am Samstag?
Die Kinder spielten im Garten, als es anfing zu schneien.""",
"pt": """O governo anunciou uma nova política econômica na segunda-feira.
Os cientistas descobriram uma nova espécie de pássaro na floresta.
A reunião será realizada na universidade na próxima semana.
Ela trabalha no hospital há muitos anos.
A previsão do tempo indica chuvas fortes durante todo o fim de semana.
As crianças deveriam ler livros todos os dias antes de dormir.
A empresa registrou um forte crescimento durante o terceiro trimestre.
Queremos agradecer a todos os que ajudaram neste projeto.
O museu está aberto das nove da manhã até as cinco da tarde.
Ele comprou uma casa pequena perto do rio com o irmão.
O transporte público da cidade melhorou muito.
Por favor, lembre-se de trazer o seu passaporte e o seu bilhete.
O presidente falará hoje à noite à nação sobre a crise.
Muitas pessoas acreditam que a educação é a chave para um futuro melhor.
O trem para o aeroporto sai a cada trinta minutos.
Nossa equipe ganhou o campeonato depois de uma temporada muito difícil.
O restaurante serve peixe fresco e legumes do mercado local.
Acho que este é o melhor livro que já li na minha vida.
Os estudantes estão se preparando para os exames finais na biblioteca.
Havia muito trânsito na estrada hoje de manhã.
Onde posso encontrar um bom médico neste bairro?
A nova lei protegerá os direitos dos trabalhadores e de suas famílias.
Minha avó contava histórias maravilhosas sobre a sua infância na aldeia.
O preço do petróleo subiu outra vez este mês.
Você quer uma xícara de café ou um chá com leite?
Os pesquisadores estudam como o clima está mudando no mundo inteiro.
O filme foi exibido pela primeira vez ontem no festival.
Eles decidiram viajar pelas montanhas durante as férias de verão.
A que horas a loja fecha aos sábados?
As crianças brincavam no jardim quando começou a nevar.""",
"it": """Il governo ha annunciato lunedì una nuova politica economica.
Gli scienziati hanno scoperto una nuova specie di uccello nella foresta.
La riunione si terrà all'università la prossima settimana.
Lei lavora in ospedale da molti anni.
Le previsioni del tempo annunciano forti piogge per tutto il fine settimana.
I bambini dovrebbero leggere libri ogni giorno prima di dormire.
L'azienda ha registrato una forte crescita durante il terzo trimestre.
Vogliamo ringraziare tutti quelli che hanno aiutato con questo progetto.
Il museo è aperto dalle nove del mattino alle cinque del pomeriggio.
Ha comprato una piccola casa vicino al fiume con suo fratello.
Il trasporto pubblico della città è migliorato molto.
Per favore, ricordati di portare il passaporto e il biglietto.
Il presidente parlerà stasera alla nazione della crisi.
Molte persone credono che l'istruzione sia la chiave per un futuro migliore.
Il treno per l'aeroporto parte ogni trenta minuti.
La nostra squadra ha vinto il campionato dopo una stagione molto difficile.
Il ristorante serve pesce fresco e verdure del mercato locale.
Penso che questo sia il miglior libro che abbia mai letto.
Gli studenti si stanno preparando per gli esami finali in biblioteca.
C'era molto traffico sull'autostrada questa mattina.
Dove posso trovare un buon medico in questo quartiere?
La nuova legge proteggerà i diritti dei lavoratori e delle loro famiglie.
Mia nonna raccontava storie meravigliose sulla sua infanzia nel paese.
Il prezzo del petrolio è aumentato di nuovo questo mese.
Vuoi una tazza di caffè o un tè con il latte?
I ricercatori studiano come sta cambiando il clima in tutto il mondo.
Il film è stato proiettato per la prima volta ieri al festival.
Hanno deciso di viaggiare attraverso le montagne durante le vacanze estive.
A che ora chiude il negozio il sabato?
I bambini giocavano in giardino quando ha cominciato a nevicare.""",
"hi": """सरकार ने सोमवार को नई आर्थिक नीति की घोषणा की।
वैज्ञानिकों ने जंगल में पक्षी की एक नई प्रजाति खोजी है।
बैठक अगले सप्ताह विश्वविद्यालय में होगी।
वह कई वर्षों से अस्पताल में काम कर रही है।
मौसम विभाग ने पूरे सप्ताहांत भारी बारिश की संभावना जताई है।
बच्चों को सोने से पहले हर दिन किताबें पढ़नी चाहिए।
कंपनी ने तीसरी तिमाही में अच्छी वृद्धि दर्ज की।
हम उन सभी को धन्यवाद देना चाहते हैं जिन्होंने इस परियोजना में मदद की।
संग्रहालय सुबह नौ बजे से शाम पाँच बजे तक खुला रहता है।
उसने अपने भाई के साथ नदी के पास एक छोटा घर खरीदा।
शहर में सार्वजनिक परिवहन काफी बेहतर हुआ है।
कृपया अपना पासपोर्ट और टिकट लाना याद रखें।
राष्ट्रपति आज रात संकट के बारे में देश को संबोधित करेंगे।
बहुत से लोग मानते हैं कि शिक्षा बेहतर भविष्य की कुंजी है।
हवाई अड्डे के लिए ट्रेन हर तीस मिनट में चलती है।
हमारी टीम ने एक बहुत कठिन सीज़न के बाद चैंपियनशिप जीती।
रेस्तरां में स्थानीय बाज़ार की ताज़ी मछली और सब्ज़ियाँ मिलती हैं।
मुझे लगता है कि यह मेरी पढ़ी हुई सबसे अच्छी किताब है।
छात्र पुस्तकालय में अपनी अंतिम परीक्षा की तैयारी कर रहे हैं।
आज सुबह राजमार्ग पर बहुत ट्रैफ़िक था।
इस मोहल्ले में मुझे अच्छा डॉक्टर कहाँ मिल सकता है?
नया कानून मज़दूरों और उनके परिवारों के अधिकारों की रक्षा करेगा।
मेरी दादी गाँव में अपने बचपन की अद्भुत कहानियाँ सुनाती थीं।
इस महीने तेल की कीमत फिर बढ़ गई है।
क्या आप एक कप कॉफ़ी या दूध वाली चाय लेंगे?
शोधकर्ता अध्ययन कर रहे हैं कि दुनिया भर में जलवायु कैसे बदल रही है।
फ़िल्म कल पहली बार महोत्सव में दिखाई गई।
उन्होंने गर्मी की छुट्टियों में पहाड़ों की यात्रा करने का फैसला किया।
शनिवार को दुकान कितने बजे बंद होती है?
जब बर्फ़ गिरने लगी तब बच्चे बगीचे में खेल रहे थे।""",
"hi-Latn": """sarkar ne somvar ko nayi economic policy ka elaan kiya
scientists ne jungle mein ek nayi chidiya ki prajati dhundhi hai
meeting agle hafte university mein hogi
woh kai saalon se hospital mein kaam kar rahi hai
mausam vibhag ne poore weekend bhaari baarish ki sambhavna batayi hai
bachchon ko sone se pehle har din kitaabein padhni chahiye
company ne teesri timahi mein acchi growth darj ki
hum un sabhi ko dhanyavaad dena chahte hain jinhone is project mein madad ki
museum subah nau baje se shaam paanch baje tak khula rehta hai
usne apne bhai ke saath nadi ke paas ek chhota ghar kharida
shehar mein public transport kaafi behtar hua hai
kripya apna passport aur ticket laana yaad rakhein
rashtrapati aaj raat sankat ke baare mein desh ko sambodhit karenge
bahut se log maante hain ki shiksha behtar bhavishya ki kunji hai
airport ke liye train har tees minute mein chalti hai
hamari team ne ek bahut mushkil season ke baad championship jeeti
restaurant mein local bazaar ki taazi machhli aur sabziyan milti hain
mujhe lagta hai ki yeh meri padhi hui sabse acchi kitaab hai
students library mein apni final pariksha ki taiyari kar rahe hain
aaj subah highway par bahut traffic tha
is mohalle mein mujhe accha doctor kahan mil sakta hai
naya kanoon mazdooron aur unke parivaron ke adhikaron ki raksha karega
meri daadi gaon mein apne bachpan ki kahaniyan sunati thi
is mahine tel ki keemat phir badh gayi hai
kya aap ek cup coffee ya doodh wali chai lenge
researchers study kar rahe hain ki duniya bhar mein climate kaise badal raha hai
film kal pehli baar festival mein dikhayi gayi
unhone garmi ki chhuttiyon mein pahadon ki yatra karne ka faisla kiya
shanivar ko dukaan kitne baje band hoti hai
jab barf girne lagi tab bachche bagiche mein khel rahe the""",
"ru": """Правительство объявило о новой экономической политике в понедельник.
Учёные обнаружили в лесу новый вид птиц.
Встреча состоится в университете на следующей неделе.
Она работает в больнице уже много лет.
Синоптики обещают сильные дожди на все выходные.
Дети должны читать книги каждый день перед сном.
Компания сообщила о значительном росте в третьем квартале.
Мы хотим поблагодарить всех, кто помогал с этим проектом.
Музей открыт с девяти утра до пяти вечера.
Он купил маленький дом у реки вместе со своим братом.
Общественный транспорт в городе стал намного лучше.
Пожалуйста, не забудьте взять паспорт и билет.
Президент сегодня вечером обратится к народу по поводу кризиса.
Многие люди считают, что образование — это ключ к лучшему будущему.
Поезд в аэропорт отправляется каждые тридцать минут.
Наша команда выиграла чемпионат после очень трудного сезона.
В ресторане подают свежую рыбу и овощи с местного рынка.
Я думаю, что это лучшая книга, которую я когда-либо читал.
Студенты готовятся к выпускным экзаменам в библиотеке.
Сегодня утром на шоссе было много машин.
Где я могу найти хорошего врача в этом районе?
Новый закон защитит права рабочих и их семей.
Моя бабушка рассказывала чудесные истории о своём детстве в деревне.
Цена на нефть в этом месяце снова выросла.
Хотите чашку кофе или чай с молоком?
Исследователи изучают, как меняется климат во всём мире.
Фильм впервые показали вчера на фестивале.
Они решили путешествовать по горам во время летних каникул.
Во сколько закрывается магазин по субботам?
Дети играли в саду, когда пошёл снег.""",
"uk": """Уряд оголосив про нову економічну політику в понеділок.
Науковці виявили в лісі новий вид птахів.
Зустріч відбудеться в університеті наступного тижня.
Вона працює в лікарні вже багато років.
Синоптики обіцяють сильні дощі на всі вихідні.
Діти повинні читати книжки щодня перед сном.
Компанія повідомила про значне зростання у третьому кварталі.
Ми хочемо подякувати всім, хто допомагав із цим проєктом.
Музей відкритий з дев'ятої ранку до п'ятої вечора.
Він купив маленький будинок біля річки разом зі своїм братом.
Громадський транспорт у місті став набагато кращим.
Будь ласка, не забудьте взяти паспорт і квиток.
Президент сьогодні ввечері звернеться до народу з приводу кризи.
Багато людей вважають, що освіта — це ключ до кращого майбутнього.
Потяг до аеропорту вирушає кожні тридцять хвилин.
Наша команда виграла чемпіонат після дуже важкого сезону.
У ресторані подають свіжу рибу та овочі з місцевого ринку.
Я думаю, що це найкраща книга, яку я коли-небудь читав.
Студенти готуються до випускних іспитів у бібліотеці.
Сьогодні вранці на шосе було багато машин.
Де я можу знайти хорошого лікаря в цьому районі?
Новий закон захистить права робітників та їхніх сімей.
Моя бабуся розповідала чудові історії про своє дитинство в селі.
Ціна на нафту цього місяця знову зросла.
Хочете чашку кави чи чай з молоком?
Дослідники вивчають, як змінюється клімат у всьому світі.
Фільм уперше показали вчора на фестивалі.
Вони вирішили подорожувати горами під час літніх канікул.
О котрій годині зачиняється магазин по суботах?
Діти гралися в саду, коли пішов сніг.""",
"ar": """أعلنت الحكومة عن سياسة اقتصادية جديدة يوم الاثنين.
اكتشف العلماء نوعاً جديداً من الطيور في الغابة.
سيعقد الاجتماع في الجامعة الأسبوع المقبل.
هي تعمل في المستشفى منذ سنوات عديدة.
تتوقع الأرصاد الجوية أمطاراً غزيرة طوال عطلة نهاية الأسبوع.
يجب على الأطفال قراءة الكتب كل يوم قبل النوم.
سجلت الشركة نمواً قوياً خلال الربع الثالث.
نود أن نشكر كل من ساعد في هذا المشروع.
المتحف مفتوح من التاسعة صباحاً حتى الخامسة مساءً.
اشترى بيتاً صغيراً قرب النهر مع أخيه.
تحسنت وسائل النقل العام في المدينة كثيراً.
من فضلك تذكر أن تحضر جواز سفرك وتذكرتك.
سيتحدث الرئيس الليلة إلى الأمة عن الأزمة.
يعتقد كثير من الناس أن التعليم هو مفتاح مستقبل أفضل.
يغادر القطار إلى المطار كل ثلاثين دقيقة.
فاز فريقنا بالبطولة بعد موسم صعب جداً.
يقدم المطعم السمك الطازج والخضروات من السوق المحلي.
أعتقد أن هذا أفضل كتاب قرأته في حياتي.
يستعد الطلاب لامتحاناتهم النهائية في المكتبة.
كان هناك ازدحام شديد على الطريق السريع هذا الصباح.
أين يمكنني أن أجد طبيباً جيداً في هذا الحي؟
سيحمي القانون الجديد حقوق العمال وعائلاتهم.
كانت جدتي تروي قصصاً رائعة عن طفولتها في القرية.
ارتفع سعر النفط مرة أخرى هذا الشهر.
هل تريد فنجان قهوة أم شاياً بالحليب؟
يدرس الباحثون كيف يتغير المناخ في جميع أنحاء العالم.
عُرض الفيلم لأول مرة أمس في المهرجان.
قرروا السفر عبر الجبال خلال العطلة الصيفية.
متى يغلق المتجر يوم السبت؟
كان الأطفال يلعبون في الحديقة عندما بدأ الثلج يتساقط.""",
"ja": """政府は月曜日に新しい経済政策を発表しました。
科学者たちは森で新しい種類の鳥を発見しました。
会議は来週、大学で行われます。
彼女は長年病院で働いています。
天気予報によると、週末はずっと大雨になるそうです。
子供たちは毎日寝る前に本を読むべきです。
その会社は第三四半期に大きな成長を記録しました。
このプロジェクトを手伝ってくれた皆さんに感謝します。
博物館は午前九時から午後五時まで開いています。
彼は兄と一緒に川の近くに小さな家を買いました。
この街の公共交通機関はかなり良くなりました。
パスポートと切符を忘れずに持ってきてください。
大統領は今夜、危機について国民に話します。
多くの人は教育がより良い未来への鍵だと信じています。
空港行きの電車は三十分ごとに出発します。
私たちのチームはとても難しいシーズンの後で優勝しました。
そのレストランでは地元の市場の新鮮な魚と野菜が食べられます。
これは今まで読んだ中で一番いい本だと思います。
学生たちは図書館で期末試験の準備をしています。
今朝は高速道路がとても混んでいました。
この近くでいい医者はどこにいますか。
新しい法律は労働者とその家族の権利を守ります。
祖母は村での子供時代の素晴らしい話をしてくれました。
今月も石油の価格が上がりました。
コーヒーかミルクティーはいかがですか。
研究者たちは世界中で気候がどのように変化しているかを調べています。
その映画は昨日、映画祭で初めて上映されました。
彼らは夏休みに山を旅行することにしました。
土曜日は何時に店が閉まりますか。
雪が降り始めたとき、子供たちは庭で遊んでいました。""",
}
FORMAL = {l: [s.strip() for s in t.split("\n") if s.strip()] for l, t in FORMAL.items()}

# Frequent words / greetings. Training only. Real LID corpora are dominated by function
# words, and they are exactly what short posts ("ok thanks", "hola q tal") consist of.
LEXICON = {
"en": "hello hi hey thanks thank you please yes no good morning night how are you what is your name i am fine love my friend today tomorrow yesterday where when why because very much the and but with for from this that have has was were will would can could should they them their there here about people time day year work home happy sorry great really just like know think want need going been don't it's i'm you're not all some more see call need so too awesome nice",
"es": "hola gracias por favor sí no buenos días buenas noches cómo estás qué tal me llamo estoy bien te quiero mi amigo hoy mañana ayer dónde cuándo por qué porque muy mucho el la los las y pero con para de este esta tengo tiene era fueron será puede debería ellos aquí allí sobre gente tiempo día año trabajo casa feliz lo siento genial realmente solo como sé pienso quiero necesito voy también nada todo más ahora siempre nunca vamos estoy eres somos hacer puedo ir",
"fr": "bonjour salut merci s'il vous plaît oui non bonne nuit bonsoir comment ça va comment allez-vous je m'appelle je vais bien je t'aime mon ami aujourd'hui demain hier où quand pourquoi parce que très beaucoup le la les et mais avec pour de ce cette j'ai il a était seront peut devrait ils elles ici là sur les gens temps jour année travail maison heureux désolé super vraiment seulement comme je sais je pense je veux j'ai besoin je vais aussi rien tout plus maintenant toujours jamais nous sommes vous êtes faire c'est qu'est-ce que je suis",
"de": "hallo guten morgen gute nacht danke bitte ja nein wie geht es dir wie heißt du ich heiße mir geht es gut ich liebe dich mein freund heute morgen gestern wo wann warum weil sehr viel der die das und aber mit für von dieser diese ich habe er hat war waren wird kann sollte sie hier dort über leute zeit tag jahr arbeit zuhause glücklich entschuldigung toll wirklich nur wie ich weiß ich denke ich will ich brauche ich gehe auch nichts alles mehr jetzt immer nie wir sind ihr seid machen nicht kein schön wir sehen uns",
"pt": "olá oi obrigado obrigada por favor sim não bom dia boa noite como vai tudo bem me chamo estou bem eu te amo meu amigo hoje amanhã ontem onde quando por que porque muito o a os as e mas com para de este esta tenho tem era foram será pode deveria eles aqui ali sobre pessoas tempo dia ano trabalho casa feliz desculpa legal realmente só como sei penso quero preciso vou também nada tudo mais agora sempre nunca vamos estou você somos fazer então não é uma um",
"it": "ciao buongiorno buonanotte grazie per favore sì no come stai come ti chiami mi chiamo sto bene ti amo il mio amico oggi domani ieri dove quando perché molto il lo la gli le e ma con per di questo questa ho ha era erano sarà può dovrebbe loro qui lì sopra gente tempo giorno anno lavoro casa felice mi dispiace fantastico davvero solo come so penso voglio ho bisogno vado anche niente tutto più adesso sempre mai andiamo sono sei siamo fare che cosa una un non c'è ragazzi",
"hi-Latn": "namaste kaise ho aap kya haal hai main theek hoon dhanyavaad shukriya haan nahi accha theek hai mera naam hai tum kahan ho kya kar rahe ho mujhe pata nahi yaar bhai dost aaj kal abhi kyun kyunki bahut zyada thoda aur lekin ke saath ke liye yeh woh hai hain tha thi hoga sakta chahiye hum tum log samay din saal kaam ghar khush maaf karo sach mein bas jaise pata hai sochta hoon chahta hoon ja raha hoon bhi kuch nahi sab kuch ab hamesha kabhi nahi chalo kya baat hai bilkul pyaar zindagi dil maza aaya",
"ru": "привет здравствуйте спасибо пожалуйста да нет доброе утро спокойной ночи как дела как тебя зовут меня зовут у меня всё хорошо я тебя люблю мой друг сегодня завтра вчера где когда почему потому что очень много и но с для от этот эта у меня есть был были будет может должен они здесь там о люди время день год работа дом счастлив извините отлично действительно только как я знаю я думаю я хочу мне нужно я иду тоже ничего всё больше сейчас всегда никогда мы вы делать что это не ещё уже просто",
"uk": "привіт добрий день дякую будь ласка так ні доброго ранку на добраніч як справи як тебе звати мене звати у мене все добре я тебе кохаю мій друг сьогодні завтра вчора де коли чому тому що дуже багато і але з для від цей ця у мене є був були буде може повинен вони тут там про люди час день рік робота дім щасливий вибачте чудово справді тільки як я знаю я думаю я хочу мені потрібно я йду теж нічого все більше зараз завжди ніколи ми ви робити що це не ще вже просто",
"hi": "नमस्ते आप कैसे हैं क्या हाल है मैं ठीक हूँ धन्यवाद शुक्रिया हाँ नहीं अच्छा मेरा नाम तुम कहाँ हो मुझे पता नहीं दोस्त भाई आज कल अभी क्यों क्योंकि बहुत ज़्यादा थोड़ा और लेकिन के साथ के लिए यह वह है हैं था थी होगा चाहिए हम लोग समय दिन साल काम घर खुश माफ़ करना सच में बस प्यार ज़िंदगी दिल",
"ar": "مرحبا شكرا من فضلك نعم لا صباح الخير تصبح على خير كيف حالك ما اسمك اسمي أنا بخير أحبك صديقي اليوم غدا أمس أين متى لماذا لأن جدا كثيرا و لكن مع من هذا هذه عندي كان سوف يمكن هم هنا هناك عن الناس وقت يوم سنة عمل بيت سعيد آسف رائع حقا فقط أعرف أريد أيضا كل شيء الآن دائما أبدا نحن أنتم ماذا ليس",
"ja": "こんにちは ありがとう おはようございます おやすみなさい はい いいえ お元気ですか 私の名前は 元気です 大好き 友達 今日 明日 昨日 どこ いつ なぜ とても たくさん これ それ あります でした できます ここ そこ 人々 時間 仕事 家 嬉しい ごめんなさい すごい 本当に だけ 知っています 思います 欲しい 行きます 何も 全部 もっと 今 いつも 絶対に 私たち あなた する じゃない ですね でしょう",
}
LEXICON = {l: t.split() for l, t in LEXICON.items()}

# Slangy social posts: abbreviations, dropped accents, dialect. First half of each list
# may be used for training (in-domain labels), second half is ALWAYS held out.
SOCIAL = {
"en": ["omg cant believe its monday again lol", "ngl that movie was kinda mid tbh", "who else is up rn?? cant sleep smh",
       "this pizza slaps fr fr", "gonna hit the gym later, wanna come?", "lmaooo my cat just knocked over my coffee",
       "idk what to wear tonight help", "soooo tired but gotta finish this hw", "u guys seen the new trailer yet?? its fire",
       "thx for the bday wishes everyone ily"],
"es": ["jajaja no manches wey q risa", "tqm amiga nos vemos al rato", "q onda, vamos x unas chelas o q?",
       "nooo no puede ser, otra vez lunes", "ando bien cansado pero ni modo, a chambear", "oye tio esta peli mola un monton",
       "xq nadie me aviso de la fiesta??", "buenisimo el partido de ayer, golazo", "tengo un hambre q me muero jaja",
       "feliz cumple bro, pasala chido"],
"fr": ["mdr t'as vu sa tête ptdr", "jsuis trop fatigué là, jpp", "wesh bien ou quoi frérot",
       "c trop bien ce film, jkiffe grave", "tkt pas on se voit dmain", "bcp trop de taf cette semaine, la flemme",
       "slt ça va ? tu fais koi ce soir", "pk personne m'a dit pour la soirée ??", "oklm à la maison, trop bien",
       "bon anniv mon pote, profite bien"],
"de": ["hahaha digga das war so krass", "hab kein bock mehr auf montag ey", "was geht ab alter, alles klar bei dir?",
       "bin sooo müde, brauch erstmal nen kaffee", "mega geil der film, musst du gucken", "kp was ich heute anziehen soll",
       "vllt gehen wir später noch was trinken", "lg und bis morgen, hdl", "alles gute zum geburtstag, feier schön",
       "boah ne schon wieder regen, nervt echt"],
"pt": ["kkkkk mano q isso, morri de rir", "vc viu o jogo ontem? foi top demais", "to mt cansado hj, so quero dormir",
       "blz, a gente se ve amanha entao", "nossa q saudade de vcs, bjs", "pq ninguem me avisou da festa??",
       "partiu praia galera, bora", "tmj irmao, parabens pelo niver", "eita q fome, alguem quer pizza?",
       "nao acredito q ja e segunda de novo aff"],
"it": ["ahahah raga che ridere, troppo forte", "cmq nn ho capito niente del film boh", "oggi sono stanchissimo, nn ce la faccio",
       "dai raga ci vediamo dopo x una birra", "ke bello sto posto, tvb", "xke nessuno mi ha detto della festa??",
       "mamma mia che fame, pizza stasera?", "auguri fra, buon compleanno", "figo sto pezzo, lo ascolto sempre",
       "uffa di nuovo lunedi, che palle"],
"hi": ["अरे यार आज तो मज़ा आ गया", "क्या बात है भाई, एकदम झकास", "फिर से सोमवार आ गया यार, बोर हो गया",
       "भूख लगी है, कोई पिज़्ज़ा मंगाओ ना", "जन्मदिन मुबारक हो भाई, खूब मस्ती कर", "नींद नहीं आ रही, कोई जगा है क्या",
       "ये फ़िल्म तो कमाल की थी सच्ची", "चल ना यार कहीं घूमने चलते हैं"],
"hi-Latn": ["arre yaar aaj toh maza aa gaya", "kya scene hai bhai, full on masti", "bhai ye movie ekdum mast thi, must watch",
            "mujhe neend nahi aa rahi, koi jaga hai kya", "happy birthday bhai, party kab de raha hai",
            "aaj office mein bahut kaam tha yaar, thak gaya", "kal milte hain phir, pakka aana",
            "abe kya bol raha hai tu, pagal hai kya", "bhook lagi hai, pizza order kar na",
            "kuch nahi yaar bas timepass chal raha hai"],
"ru": ["ахах ну ты даёшь, ржу не могу", "блин опять понедельник, не хочу на работу", "спс за поздравления, всех люблю",
       "чё делаешь сегодня вечером? го гулять", "кароч фильм огонь, всем советую", "щас приду, пять сек",
       "ваще не понял о чём речь лол", "с днюхой бро, всего самого", "норм так посидели вчера, надо повторить",
       "жрать хочу, закажем пиццу?"],
"uk": ["ахах ну ти даєш, не можу", "блін знову понеділок, не хочу на роботу", "дяка за привітання, всіх люблю",
       "шо робиш сьогодні ввечері? гайда гуляти", "коротше фільм вогонь, всім раджу", "зара прийду, п'ять сек",
       "взагалі не зрозумів про шо мова", "з днюхою бро, всього найкращого", "класно посиділи вчора, треба повторити",
       "їсти хочу, замовимо піцу?"],
"ar": ["هههه والله ضحكتني يا رجل", "شو الأخبار يا شباب، وين السهرة اليوم", "يا ربي مرة ثانية يوم الاثنين، تعبت",
       "مبروك يا صاحبي، كل عام وانت بخير", "جوعان كتير، مين بدو بيتزا", "الفيلم كان حلو كتير، لازم تشوفوه",
       "مش قادر أنام، حدا صاحي", "يلا نطلع نتمشى شوي"],
"ja": ["まじで眠いんだけど笑", "今日のライブやばかったwww", "また月曜日か、だるいなあ", "お腹すいた、誰かピザ食べない？",
       "誕生日おめでとう！楽しんでね", "この映画めっちゃ良かった、おすすめ", "寝れない、誰か起きてる？", "それなすぎて草"],
}

# Code-mixed posts with the SET of languages present (multi-label ground truth).
CODEMIX = [
    ("I'll call you mañana porque tengo mucho trabajo today", ["en", "es"]),
    ("yaar this movie was awesome, bahut maza aaya sach mein", ["en", "hi-Latn"]),
    ("मुझे ये song बहुत पसंद है, really nice lyrics", ["hi", "en"]),
    ("je suis so tired aujourd'hui, I need a coffee", ["fr", "en"]),
    ("это просто amazing, спасибо my friend", ["ru", "en"]),
    ("no puedo ir to the party tonight, lo siento mucho", ["es", "en"]),
    ("wir sehen uns morgen at the meeting, see you there", ["de", "en"]),
    ("kal office mein meeting hai, please be on time everyone", ["hi-Latn", "en"]),
    ("今日は very busy でした、I am so tired", ["ja", "en"]),
    ("obrigado my friend, você é the best", ["pt", "en"]),
    ("ciao ragazzi, see you tomorrow at the restaurant", ["it", "en"]),
    ("شكرا يا صديقي, thank you so much for everything", ["ar", "en"]),
]

# Out-of-distribution posts: languages the model was never trained on + junk.
# The right answer for all of them is "unknown" (open-set recognition).
OOD = [
    ("Ik weet niet waar mijn fiets is gebleven vanochtend", "Dutch"),
    ("Bugün hava çok güzel, dışarı çıkmak istiyorum", "Turkish"),
    ("Jag skulle vilja ha en kopp kaffe, tack så mycket", "Swedish"),
    ("Nie wiem, gdzie jest mój klucz do mieszkania", "Polish"),
    ("Saya tidak tahu di mana kunci rumah saya", "Indonesian"),
    ("Hôm nay trời đẹp quá, tôi muốn đi dạo", "Vietnamese"),
    ("Minä en tiedä missä avaimeni ovat tänään", "Finnish"),
    ("Nem tudom, hol van a kulcsom ma reggel", "Hungarian"),
    ("오늘 날씨가 정말 좋네요, 산책하러 갈까요", "Korean (unsupported script)"),
    ("Καλημέρα, πώς είσαι σήμερα φίλε μου", "Greek (unsupported script)"),
    ("วันนี้อากาศดีมาก ฉันอยากออกไปเดินเล่น", "Thai (unsupported script)"),
    ("היום מזג האוויר יפה מאוד", "Hebrew (unsupported script)"),
    ("asdkjh qwlekj zxmcnv poiuqw", "keyboard mash"),
    ("12345 67890 :) :) !!!", "no letters"),
]

TEST_SENT = [2, 6, 9, 13, 17, 21, 25, 28]          # same held-out sentence ids in every language
TRAIN_SENT = [i for i in range(30) if i not in TEST_SENT]
_PUNCT = " .,;:!?¿¡。、？！،؟—\"'"

# =====================================================================================
# 2. PREPROCESSING + NOISE (interview concept: URLs / @mentions / #hashtags / emoji are
#    not language; hashtags and handles are mostly English regardless of the post)
# =====================================================================================
_RE_URL = re.compile(r"(https?://\S+|www\.\S+)")
_RE_MENTION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_RE_DIGIT = re.compile(r"\d+")
_RE_ELONG = re.compile(r"(.)\1{2,}")
_RE_WS = re.compile(r"\s+")


def clean(text: str) -> tuple[str, dict]:
    """Strip non-linguistic tokens. Returns (cleaned_text, counts of what was removed)."""
    removed = {}
    for name, rx in (("urls", _RE_URL), ("mentions", _RE_MENTION), ("hashtags", _RE_HASHTAG), ("emoji", _RE_EMOJI)):
        text, n = rx.subn(" ", text)
        removed[name] = n
    text = _RE_DIGIT.sub(" ", text)
    text = _RE_ELONG.sub(r"\1\1", text)          # "holaaaaa" -> "holaa"
    return _RE_WS.sub(" ", text).strip(), removed


_HANDLES = ["@john_doe92", "@the_real_maria", "@official_news", "@best_memes_daily", "@travel_with_me", "@mike_and_kate"]
_TAGS = ["#love", "#instagood", "#photooftheday", "#follow", "#nofilter", "#happy", "#tbt", "#weekendvibes", "#mondaymotivation"]
_EMOJI = ["\U0001F602", "\U0001F525", "❤️", "\U0001F62D", "\U0001F64F", "\U0001F389"]
_ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def add_noise(text: str, rng, kinds=("url", "mention", "hashtag", "emoji")) -> str:
    """Wrap a clean post in typical social-media decoration."""
    pre, post = [], []
    if "mention" in kinds:
        pre.append(_HANDLES[rng.integers(len(_HANDLES))])
    if "emoji" in kinds:
        post.append("".join(_EMOJI[rng.integers(len(_EMOJI))] for _ in range(int(rng.integers(1, 4)))))
    if "hashtag" in kinds:
        post += [_TAGS[i] for i in rng.choice(len(_TAGS), size=int(rng.integers(1, 4)), replace=False)]
    if "url" in kinds:
        slug = "".join(_ALNUM[i] for i in rng.integers(0, len(_ALNUM), 10))
        post.append(("https://t.co/" if rng.random() < 0.5 else "https://www.youtube.com/watch?v=") + slug)
    return " ".join(pre + [text] + post)


def strip_accents(text: str) -> str:
    """Latin-only accent stripping: people type 'manana', 'tres bien', 'nao' on phones."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# =====================================================================================
# 3. STAGE 1 - SCRIPT RULE (interview concept: cheap deterministic stage in a cascade.
#    Unicode block decides the language outright for Devanagari/Arabic/Kana here and
#    narrows the candidate set to 2 for Cyrillic and 7 for Latin.)
# =====================================================================================
_RANGES = [
    (0x41, 0x5A, "Latin"), (0x61, 0x7A, "Latin"), (0xC0, 0x24F, "Latin"), (0x1E00, 0x1EFF, "Latin"),
    (0x370, 0x3FF, "Greek"), (0x400, 0x52F, "Cyrillic"), (0x590, 0x5FF, "Hebrew"),
    (0x600, 0x6FF, "Arabic"), (0x750, 0x77F, "Arabic"), (0x900, 0x97F, "Devanagari"),
    (0xE00, 0xE7F, "Thai"), (0x1100, 0x11FF, "Hangul"), (0x3040, 0x30FF, "CJK"),
    (0x4E00, 0x9FFF, "CJK"), (0xAC00, 0xD7AF, "Hangul"),
]
# NOTE: with Chinese in scope, Han-only text would be {zh, ja} and kana would decide ja.
SCRIPT_LANGS = {
    "Latin": ["en", "es", "fr", "de", "pt", "it", "hi-Latn"], "Cyrillic": ["ru", "uk"],
    "Arabic": ["ar"], "Devanagari": ["hi"], "CJK": ["ja"],
    "Greek": [], "Hebrew": [], "Thai": [], "Hangul": [],
}


def char_script(ch: str) -> str | None:
    cp = ord(ch)
    if cp in (0xD7, 0xF7):
        return None
    for lo, hi, name in _RANGES:
        if lo <= cp <= hi:
            return name
    return None


def script_profile(text: str) -> dict:
    counts: dict[str, int] = {}
    for ch in text:
        s = char_script(ch)
        if s:
            counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    if not total:
        return {"dominant": None, "shares": {}, "candidates": [], "mixed": False, "letters": 0}
    shares = {k: round(v / total, 3) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}
    dom = next(iter(shares))
    return {"dominant": dom, "shares": shares, "candidates": SCRIPT_LANGS.get(dom, []),
            "mixed": len(shares) > 1 and shares[dom] < 0.9, "letters": total}


# =====================================================================================
# 4. SYNTHETIC SAMPLES (windows over sentences, shuffled lexicon bags, augmentation)
# =====================================================================================
def _window(sent: str, lang: str, rng, min_units=1, max_units=None) -> str:
    if lang in NO_SPACE:
        s = sent.strip(_PUNCT)
        lo = min(max(3, min_units * 2), len(s))
        n = int(rng.integers(lo, (min(len(s), max_units * 3) if max_units else len(s)) + 1))
        st = int(rng.integers(0, len(s) - n + 1))
        return s[st:st + n].strip(_PUNCT)
    w = sent.split()
    lo = min(min_units, len(w))
    hi = min(len(w), max_units) if max_units else len(w)
    n = int(rng.integers(lo, max(lo, hi) + 1))
    st = int(rng.integers(0, len(w) - n + 1))
    return " ".join(w[st:st + n]).strip(_PUNCT)


def _augment(text: str, lang: str, rng) -> str:
    """Make formal text look like a phone-typed post (domain-shift mitigation)."""
    if lang in ("es", "fr", "de", "pt", "it") and rng.random() < 0.7:
        text = strip_accents(text)
    text = re.sub(r"[.,;:!?¿¡]", "", text).lower()
    if lang not in NO_SPACE and rng.random() < 0.3 and len(text) > 3:   # elongation: "siii", "nooo"
        i = int(rng.integers(1, len(text)))
        if text[i].isalpha():
            text = text[:i] + text[i] * int(rng.integers(2, 4)) + text[i + 1:]
    return text


def _synth(lang: str, sent_ids, n: int, rng, min_units=1, max_units=None, double=0.0) -> list[tuple[str, str, int]]:
    """n samples (text, lang, source_sentence_id) built from windows over the given sentences."""
    out = []
    for _ in range(n):
        sid = int(sent_ids[rng.integers(len(sent_ids))])
        t = _window(FORMAL[lang][sid], lang, rng, min_units, max_units)
        if double and rng.random() < double:
            sid2 = int(sent_ids[rng.integers(len(sent_ids))])
            t = t + (" " if lang not in NO_SPACE else "、") + _window(FORMAL[lang][sid2], lang, rng, min_units, max_units)
        out.append((t, lang, sid))
    return out


def _lexicon_bags(lang: str, n: int, rng) -> list[tuple[str, str, int]]:
    lex = LEXICON[lang]
    out = []
    for _ in range(n):
        k = int(rng.integers(1, 6))
        words = [lex[i] for i in rng.integers(0, len(lex), k)]
        out.append(((" " if lang not in NO_SPACE else "").join(words), lang, -1))
    return out


def _social_split(lang: str, part: str) -> list[str]:
    posts = SOCIAL[lang]
    h = len(posts) // 2
    return posts[:h] if part == "train" else posts[h:]


def _social_samples(lang: str, part: str, rng, n_windows=2) -> list[tuple[str, str, int]]:
    out = []
    for p in _social_split(lang, part):
        out.append((p, lang, -2))
        for _ in range(n_windows):
            if lang in NO_SPACE:
                n = int(rng.integers(min(4, len(p)), len(p) + 1)); st = int(rng.integers(0, len(p) - n + 1))
                out.append((p[st:st + n], lang, -2))
            else:
                w = p.split(); n = int(rng.integers(min(3, len(w)), len(w) + 1)); st = int(rng.integers(0, len(w) - n + 1))
                out.append((" ".join(w[st:st + n]), lang, -2))
    return out


def make_train(seed=7, per_lang=170, lex=60, formal_only=False, counts: dict | None = None):
    """Training set. formal_only=True is the 'wrong way' for the domain-shift pitfall."""
    rng = np.random.default_rng(seed)
    rows = []
    for lang in LANGS:
        part = _synth(lang, TRAIN_SENT, per_lang, rng, double=0.2) + _lexicon_bags(lang, lex, rng)
        if not formal_only:
            base = _synth(lang, TRAIN_SENT, 50, rng, double=0.2)
            part += [(_augment(t, lang, rng), l, s) for t, l, s in base]         # style augmentation
            part += _social_samples(lang, "train", rng)                           # a few in-domain labels
        if counts and lang in counts:                                             # class-imbalance experiment
            idx = rng.permutation(len(part))[:counts[lang]]
            part = [part[i] for i in sorted(idx)]
        rows += part
    return rows


def make_test(seed=11, per_lang=90):
    rng = np.random.default_rng(seed)
    rows = []
    for lang in LANGS:
        seen, part = set(), []
        for t, l, s in _synth(lang, TEST_SENT, per_lang * 2, rng, min_units=2):
            if t not in seen and len(part) < per_lang:
                seen.add(t); part.append((t, l, s))
        rows += part
    return rows


def make_social_test(seed=13):
    rng = np.random.default_rng(seed)
    rows = []
    for lang in LANGS:
        rows += _social_samples(lang, "test", rng, n_windows=3)
    return rows


# =====================================================================================
# 5. MODELS
# =====================================================================================
def _pad(texts):
    return [f" {t} " for t in texts]           # spaces mark word boundaries: " el" != "el"


def _tfidf():
    # char 1-4 grams: 1-grams catch unique letters (ñ, ß, ã, і), 3-4-grams catch morphemes
    return TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=True, sublinear_tf=True, min_df=2, dtype=np.float32)


class HashedNgramBag:
    """fastText-style supervised classifier, written out in numpy.

    text -> char n-grams (1..4) -> hash each into one of B buckets -> average the bucket
    embeddings (dim d) -> linear layer -> softmax. No vocabulary is stored: model size is
    B*d + d*C floats no matter how many n-grams exist (the 'hashing trick').
    Interview concepts: fixed memory footprint, collisions, OOV-free, on-device friendly.
    """

    def __init__(self, n_buckets=1 << 15, dim=24, ngram=(1, 4), epochs=70, lr=0.05, seed=0):
        self.B, self.d, self.ngram, self.epochs, self.lr, self.seed = n_buckets, dim, ngram, epochs, lr, seed

    def ngrams(self, text: str) -> list[str]:
        t = f" {text.lower()} "
        return [t[i:i + n] for n in range(self.ngram[0], self.ngram[1] + 1) for i in range(len(t) - n + 1)]

    def _ids(self, text: str) -> list[int]:
        return [zlib.crc32(g.encode("utf-8")) % self.B for g in self.ngrams(text)]

    def _bag(self, texts):
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            ids = self._ids(t)
            if ids:
                rows += [r] * len(ids); cols += ids; vals += [1.0 / len(ids)] * len(ids)
        return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(len(texts), self.B))

    def fit(self, texts, y, n_classes, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        A = self._bag(texts); AT = A.T.tocsr(); n = len(texts)
        Y = np.zeros((n, n_classes), np.float32); Y[np.arange(n), y] = 1
        sw = np.ones(n, np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
        sw = (sw / sw.sum())[:, None]
        E = rng.normal(0, 0.1, (self.B, self.d)).astype(np.float32)
        W = np.zeros((self.d, n_classes), np.float32); b = np.zeros(n_classes, np.float32)
        params = [E, W, b]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
        self.loss_curve = []
        for ep in range(1, self.epochs + 1):                    # full-batch Adam
            H = A @ E                                           # mean of n-gram embeddings
            Z = H @ W + b
            Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
            self.loss_curve.append(float(-(sw[:, 0] * np.log(P[np.arange(n), y] + 1e-9)).sum()))
            G = (P - Y) * sw                                    # dLoss/dlogits
            grads = [AT @ (G @ W.T), H.T @ G, G.sum(0)]
            for p, g, mi, vi in zip(params, grads, m, v):
                mi *= 0.9; mi += 0.1 * g; vi *= 0.999; vi += 0.001 * g * g
                p -= self.lr * (mi / (1 - 0.9 ** ep)) / (np.sqrt(vi / (1 - 0.999 ** ep)) + 1e-8)
        self.E, self.W, self.b = E, W, b
        return self

    def predict_proba(self, texts):
        Z = (self._bag(texts) @ self.E) @ self.W + self.b
        Z -= Z.max(1, keepdims=True); P = np.exp(Z)
        return P / P.sum(1, keepdims=True)

    def explain(self, text: str, cls: int, k=8):
        grams = self.ngrams(text)
        if not grams:
            return []
        agg: dict[str, float] = {}
        wc = self.W[:, cls] - self.W.mean(1)
        for g in grams:
            agg[g] = agg.get(g, 0.0) + float(self.E[zlib.crc32(g.encode("utf-8")) % self.B] @ wc) / len(grams)
        return sorted(agg.items(), key=lambda kv: -kv[1])[:k]

    @property
    def size_kb(self):
        return (self.E.size + self.W.size + self.b.size) * 4 / 1024


class _State:
    pass


_LOCK = threading.Lock()
_S: _State | None = None
MODEL_NAMES = {"lr": "TF-IDF + Logistic Regression", "nb": "TF-IDF + Multinomial NB", "fasttext": "fastText-style hashed n-gram bag (numpy)"}


def _fit_tfidf_model(kind: str, X, y, class_weight=None):
    if kind == "nb":
        return MultinomialNB(alpha=0.05).fit(X, y)
    return LogisticRegression(C=20.0, max_iter=60, class_weight=class_weight).fit(X, y)


def state() -> _State:
    """Build corpora + train the three models once (lazy, thread-safe)."""
    global _S
    if _S is not None:
        return _S
    with _LOCK:
        if _S is not None:
            return _S
        t0 = time.perf_counter()
        S = _State()
        S.train = make_train()
        S.test = make_test()
        S.social = make_social_test()
        tr_x = [t for t, _, _ in S.train]; S.y_train = np.array([L_IDX[l] for _, l, _ in S.train])
        S.vec = _tfidf(); S.X_train = S.vec.fit_transform(_pad(tr_x))
        S.vocab = np.array(S.vec.get_feature_names_out())
        S.models = {"nb": _fit_tfidf_model("nb", S.X_train, S.y_train), "lr": _fit_tfidf_model("lr", S.X_train, S.y_train)}
        S.ft = HashedNgramBag().fit(tr_x, S.y_train, len(LANGS))
        S.build_s = round(time.perf_counter() - t0, 2)
        _S = S
    return _S


def model_proba(S, model: str, texts, vec=None, clf=None) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(LANGS)))
    if model == "fasttext" and clf is None:
        return S.ft.predict_proba(texts)
    return (clf or S.models[model]).predict_proba((vec or S.vec).transform(_pad(texts)))


# =====================================================================================
# 6. THE CASCADE (interview concept: stage 1 script rule answers the easy traffic for
#    ~free and restricts the label space of stage 2; abstain when nothing is supported)
# =====================================================================================
def classify(texts, model="lr", cascade=True, preprocess=True, S=None, vec=None, clf=None):
    """Returns (P [n x L] probabilities - all-zero row means 'unknown at stage 1', stage [n])."""
    S = S or state()
    cleaned = [clean(t)[0] for t in texts] if preprocess else list(texts)
    P = np.zeros((len(texts), len(LANGS)))
    stage = np.zeros(len(texts), int)        # 0 = unknown (no supported script), 1 = script rule, 2 = model
    need = []
    for i, t in enumerate(cleaned):
        prof = script_profile(t)
        if cascade:
            c = prof["candidates"]
            if len(c) == 1:
                P[i, L_IDX[c[0]]] = 1.0; stage[i] = 1
            elif len(c) > 1:
                need.append((i, c))
        elif prof["letters"]:
            need.append((i, LANGS))
    if need:
        Pm = model_proba(S, model, [cleaned[i] for i, _ in need], vec, clf)
        for row, (i, cands) in zip(Pm, need):
            mask = np.zeros(len(LANGS)); mask[[L_IDX[c] for c in cands]] = 1
            r = row * mask
            P[i] = r / r.sum() if r.sum() > 0 else mask / mask.sum()
            stage[i] = 2
    return P, stage


def metrics(y_true: np.ndarray, P: np.ndarray, threshold=0.0) -> dict:
    """Accuracy / macro-F1 / per-language P-R-F1 / confusion. 'unknown' predictions count as errors."""
    n_l = len(LANGS)
    conf = P.max(1); pred = P.argmax(1)
    pred = np.where((conf <= 0) | (conf < threshold), -1, pred)
    cm = np.zeros((n_l, n_l + 1), int)
    for t, p in zip(y_true, pred):
        cm[t, p if p >= 0 else n_l] += 1
    per = []
    for i, l in enumerate(LANGS):
        tp = cm[i, i]; sup = cm[i].sum(); pp = cm[:, i].sum()
        pr = tp / pp if pp else 0.0; rc = tp / sup if sup else 0.0
        per.append({"lang": l, "name": LANG_NAMES[l], "precision": round(pr, 3), "recall": round(rc, 3),
                    "f1": round(2 * pr * rc / (pr + rc), 3) if pr + rc else 0.0, "support": int(sup)})
    present = [p for p in per if p["support"]]
    return {"accuracy": round(float((pred == y_true).mean()), 4),
            "macro_f1": round(float(np.mean([p["f1"] for p in present])), 4),
            "coverage": round(float((pred >= 0).mean()), 4), "per_language": per, "confusion": cm.tolist()}


def ece(y_true, P, bins=10) -> float:
    """Expected calibration error: |confidence - accuracy| averaged over confidence bins."""
    conf = P.max(1); ok = P.argmax(1) == y_true
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return round(float(e), 4)


def _xy(rows):
    return [t for t, _, _ in rows], np.array([L_IDX[l] for _, l, _ in rows])


# =====================================================================================
# 7. ENDPOINTS
# =====================================================================================
@router.get("/overview")
@lru_cache(maxsize=1)
def overview():
    """Dataset stats + model comparison (accuracy, macro-F1, calibration, size, latency)."""
    S = state()
    tx, ty = _xy(S.test); sx, sy = _xy(S.social)
    rows = []
    for m in ("nb", "lr", "fasttext"):
        t0 = time.perf_counter(); P, stage = classify(tx, m, S=S); dt = time.perf_counter() - t0
        Pn, _ = classify(tx, m, cascade=False, S=S)
        Ps, _ = classify(sx, m, S=S)
        mt = metrics(ty, P)
        if m == "fasttext":
            size, n_par = S.ft.size_kb, S.ft.E.size + S.ft.W.size
        else:
            n_par = S.X_train.shape[1] * len(LANGS)
            size = (n_par * 4 + sum(len(v.encode()) + 8 for v in S.vocab)) / 1024
        rows.append({"model": m, "name": MODEL_NAMES[m], "clean_acc": mt["accuracy"], "clean_macro_f1": mt["macro_f1"],
                     "no_cascade_acc": metrics(ty, Pn)["accuracy"], "social_acc": metrics(sy, Ps)["accuracy"],
                     "ece": ece(ty[stage == 2], P[stage == 2]), "size_kb": round(size, 1), "params": int(n_par),
                     "us_per_doc": round(dt / len(tx) * 1e6, 1)})
    _, stage = classify(tx, "lr", S=S)
    counts = {l: 0 for l in LANGS}
    for _, l, _ in S.train:
        counts[l] += 1
    return {
        "languages": [{"lang": l, "name": LANG_NAMES[l], "script": next(s for s, ls in SCRIPT_LANGS.items() if l in ls),
                       "formal_sentences": len(FORMAL[l]), "lexicon_words": len(LEXICON[l]),
                       "social_posts": len(SOCIAL[l]), "train_samples": counts[l]} for l in LANGS],
        "dataset": {"languages": len(LANGS), "train_samples": len(S.train), "clean_test_samples": len(S.test),
                    "social_test_samples": len(S.social), "tfidf_features": int(S.X_train.shape[1]),
                    "held_out_sentence_ids": TEST_SENT, "codemix_samples": len(CODEMIX), "ood_samples": len(OOD),
                    "build_seconds": S.build_s},
        "cascade": {"stage1_share": round(float((stage == 1).mean()), 3), "stage2_share": round(float((stage == 2).mean()), 3),
                    "script_candidates": SCRIPT_LANGS,
                    "note": "Stage 1 (Unicode script) resolves hi/ar/ja outright; Latin (7 langs) and Cyrillic (ru/uk) go to the n-gram model."},
        "models": rows,
        "chance_accuracy": round(1 / len(LANGS), 3),
    }


class PredictIn(BaseModel):
    text: str = Field(..., max_length=2000)
    model: str = "lr"
    threshold: float = Field(0.5, ge=0, le=1)
    top_k: int = Field(5, ge=1, le=12)
    preprocess: bool = True
    cascade: bool = True


def _explain(S, model: str, text: str, cls: int, k=8):
    """Which char n-grams pushed the score towards `cls` (vs the average class)."""
    if model == "fasttext":
        return S.ft.explain(text, cls, k)
    x = S.vec.transform(_pad([text]))
    clf = S.models[model]
    w = clf.coef_ if model == "lr" else clf.feature_log_prob_
    contrib = x.data * (w[cls, x.indices] - w[:, x.indices].mean(0))
    order = np.argsort(-contrib)[:k]
    return [(S.vocab[x.indices[j]], float(contrib[j])) for j in order if contrib[j] > 0]


def _tokens(text: str) -> list[tuple[str, str | None]]:
    """Split into (token, script) units; a token never spans two scripts."""
    out, cur, cur_s = [], "", None
    for ch in text:
        s = char_script(ch)
        if s is None and not (ch in "'’-" and cur):
            if cur:
                out.append((cur, cur_s)); cur, cur_s = "", None
            continue
        if s is not None and cur and cur_s is not None and s != cur_s:
            out.append((cur, cur_s)); cur = ""
        cur += ch
        cur_s = s or cur_s
    if cur:
        out.append((cur, cur_s))
    return [(t, s) for t, s in out if s]


def segment(S, text: str, model="lr", switch_penalty=2.0):
    """Code-mixing: per-token language with a tiny Viterbi (HMM) smoother.

    emission(token, lang) = log P(lang | token +-1 same-script neighbours), masked to the
    languages the token's script allows; transition cost = switch_penalty when the
    language changes. Output = language spans (the multi-label view of a post).
    """
    toks = _tokens(text)
    if not toks:
        return [], []
    ctx = []
    for i, (t, s) in enumerate(toks):
        ctx.append(" ".join(tt for tt, ss in toks[max(0, i - 1):i + 2] if ss == s))
    Pm = model_proba(S, model, ctx)
    n, n_l = len(toks), len(LANGS)
    em = np.full((n, n_l), -1e9)
    for i, (t, s) in enumerate(toks):
        for c in SCRIPT_LANGS.get(s, []):
            em[i, L_IDX[c]] = 0.0 if len(SCRIPT_LANGS[s]) == 1 else np.log(Pm[i, L_IDX[c]] + 1e-6)
        if not SCRIPT_LANGS.get(s):
            em[i] = -20.0                                   # unsupported script: no preference
    cost = np.full((n_l, n_l), -switch_penalty); np.fill_diagonal(cost, 0.0)
    score = em[0].copy(); back = np.zeros((n, n_l), int)
    for i in range(1, n):
        cand = score[:, None] + cost
        back[i] = cand.argmax(0); score = cand.max(0) + em[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    spans = []
    for (t, s), p in zip(toks, path):
        lang = LANGS[p] if SCRIPT_LANGS.get(s) else "unknown"
        if spans and spans[-1]["lang"] == lang:
            spans[-1]["text"] += " " + t; spans[-1]["tokens"] += 1
        else:
            spans.append({"text": t, "lang": lang, "tokens": 1})
    tot = {}
    for sp in spans:
        if sp["lang"] != "unknown":
            tot[sp["lang"]] = tot.get(sp["lang"], 0) + (sp["tokens"] if sp["lang"] not in NO_SPACE else max(2, len(sp["text"]) // 2))
    n_tok = sum(tot.values()) or 1
    langs = [l for l, c in sorted(tot.items(), key=lambda kv: -kv[1]) if c >= 2 or c / n_tok >= 0.3]
    return spans, langs


@router.post("/predict")
def predict(q: PredictIn):
    """Run one post through the cascade and expose every stage."""
    if q.model not in MODEL_NAMES:
        raise HTTPException(400, f"model must be one of {list(MODEL_NAMES)}")
    S = state()
    t0 = time.perf_counter()
    cleaned, removed = clean(q.text) if q.preprocess else (q.text, {})
    t1 = time.perf_counter()
    prof = script_profile(cleaned)
    t2 = time.perf_counter()
    P, stage = classify([q.text], q.model, q.cascade, q.preprocess, S)
    t3 = time.perf_counter()
    p = P[0]; st = int(stage[0])
    order = np.argsort(-p)[:q.top_k]
    conf = float(p.max())
    abstain = conf <= 0 or conf < q.threshold
    best = int(p.argmax())
    reason = ("no letters from a supported script" if st == 0 else
              f"confidence {conf:.2f} < threshold {q.threshold:.2f}" if abstain else None)
    grams = _explain(S, q.model, cleaned, best) if st == 2 else []
    spans, langs = segment(S, cleaned, q.model if q.model != "fasttext" else "lr")
    return {
        "input": q.text, "cleaned": cleaned, "removed": removed, "model": q.model,
        "script": prof, "stage": st,
        "stage_label": {0: "stage 1: unsupported / no script -> unknown", 1: "stage 1: script rule decided, model skipped",
                        2: f"stage 2: {MODEL_NAMES[q.model]}" + (" restricted to script candidates" if q.cascade else " over all languages")}[st],
        "prediction": "unknown" if abstain else LANGS[best],
        "prediction_name": "Unknown (abstained)" if abstain else LANG_NAMES[LANGS[best]],
        "abstained": abstain, "abstain_reason": reason, "confidence": round(conf, 4),
        "top": [{"lang": LANGS[i], "name": LANG_NAMES[LANGS[i]], "prob": round(float(p[i]), 4)} for i in order if p[i] > 0],
        "ngrams": [{"ngram": g.replace(" ", "␣"), "weight": round(w, 4)} for g, w in grams],
        "segments": spans, "languages_detected": langs, "code_mixed": len(langs) > 1,
        "latency_us": {"preprocess": round((t1 - t0) * 1e6), "script_rule": round((t2 - t1) * 1e6),
                       "classify_total": round((t3 - t2) * 1e6)},
    }


@router.get("/samples")
def samples():
    """Texts for the UI's one-click examples."""
    return {
        "clean": [{"text": FORMAL[l][TEST_SENT[1]], "label": l} for l in LANGS],
        "short": [{"text": t, "label": l} for t, l in [("de nada", "es/pt"), ("no", "en/es/it"), ("ok grazie", "it"), ("la casa", "es/it"),
                                                         ("obrigado", "pt"), ("ça va", "fr"), ("привет", "ru"), ("дякую", "uk")]],
        "social": [{"text": _social_split(l, "test")[0], "label": l} for l in LANGS],
        "noisy": [{"text": "@john_doe92 gracias por todo \U0001F602\U0001F602 #love #instagood https://t.co/aB3xYz91Qk", "label": "es"},
                  {"text": "@official_news شكرا \U0001F64F #follow #happy #tbt https://www.youtube.com/watch?v=dQw4w9WgXcQ", "label": "ar"},
                  {"text": "@travel_with_me дякую #weekendvibes #nofilter https://t.co/Zk29dLmQ1x", "label": "uk"}],
        "codemix": [{"text": t, "label": "+".join(g)} for t, g in CODEMIX],
        "ood": [{"text": t, "label": n} for t, n in OOD],
    }


@lru_cache(maxsize=32)
def _evaluate(model: str, cascade: bool, test_set: str):
    S = state()
    rows = S.test if test_set == "clean" else S.social
    x, y = _xy(rows)
    P, stage = classify(x, model, cascade, True, S)
    mt = metrics(y, P)
    cm = np.array(mt["confusion"])
    pairs = [{"true": LANGS[i], "pred": LANGS[j] if j < len(LANGS) else "unknown", "count": int(cm[i, j]),
              "share_of_true": round(cm[i, j] / max(1, cm[i].sum()), 3)}
             for i in range(len(LANGS)) for j in range(len(LANGS) + 1) if i != j and cm[i, j]]
    pairs.sort(key=lambda d: -d["count"])
    # ---- abstention: sweep the confidence threshold (coverage vs selective accuracy vs OOD rejection)
    ood_x = [t for t, _ in OOD]
    Po, _ = classify(ood_x, model, cascade, True, S)
    conf = P.max(1); ok = P.argmax(1) == y; oconf = Po.max(1)
    sweep = []
    for th in np.round(np.arange(0, 1.0, 0.05), 2):
        ans = (conf >= th) & (conf > 0)
        sweep.append({"threshold": float(th), "coverage": round(float(ans.mean()), 4),
                      "selective_accuracy": round(float(ok[ans].mean()), 4) if ans.any() else 1.0,
                      "error_rate_overall": round(float((ans & ~ok).mean()), 4),
                      "ood_rejected": round(float(((oconf < th) | (oconf <= 0)).mean()), 4)})
    # ---- code-mixing: single label vs top-2 vs span-level detection
    seg_model = model if model != "fasttext" else "lr"
    hit_top2 = hit_seg = 0; cm_rows = []
    for t, gold in CODEMIX:
        Pc, _ = classify([t], model, cascade, True, S)
        top2 = [LANGS[i] for i in np.argsort(-Pc[0])[:2] if Pc[0][i] > 0.1]
        _, langs = segment(S, clean(t)[0], seg_model)
        a, b = set(gold) <= set(top2), set(gold) == set(langs)
        hit_top2 += a; hit_seg += b
        cm_rows.append({"text": t, "gold": "+".join(gold), "top2_of_single_label": "+".join(top2), "span_level": "+".join(langs), "span_exact": bool(b)})
    return {
        "model": model, "cascade": cascade, "test_set": test_set, "n": len(x), "labels": LANGS + ["unknown"],
        "accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"], "ece_stage2": ece(y[stage == 2], P[stage == 2]) if (stage == 2).any() else 0.0,
        "confusion": mt["confusion"], "per_language": mt["per_language"], "top_confusions": pairs[:8],
        "threshold_sweep": sweep,
        "ood": [{"text": t, "what": n, "predicted": LANGS[int(p.argmax())] if p.max() > 0 else "unknown", "confidence": round(float(p.max()), 3)}
                for (t, n), p in zip(OOD, Po)],
        "codemix": {"n": len(CODEMIX), "both_in_top2": round(hit_top2 / len(CODEMIX), 3),
                    "span_level_exact": round(hit_seg / len(CODEMIX), 3), "rows": cm_rows},
    }


@router.get("/evaluate")
def evaluate(model: str = "lr", cascade: bool = True, test_set: str = "clean"):
    """Confusion matrix, per-language P/R/F1, threshold sweep (abstention), OOD and code-mix eval."""
    if model not in MODEL_NAMES or test_set not in ("clean", "social"):
        raise HTTPException(400, "bad model or test_set")
    return _evaluate(model, cascade, test_set)


@router.get("/length_curve")
@lru_cache(maxsize=1)
def length_curve():
    """Accuracy vs text length (chars) for each model - short text is the hard part of LID."""
    S = state()
    buckets = [3, 5, 8, 12, 16, 24, 32, 48]
    out = {"lengths": buckets, "series": [], "n_per_bucket": []}
    sets = []
    for n in buckets:
        x, y = [], []
        for lang in LANGS:
            for sid in TEST_SENT:
                s = FORMAL[lang][sid].strip(_PUNCT)
                starts = range(0, max(1, len(s) - 2), 3) if lang in NO_SPACE else [0] + [i + 1 for i, c in enumerate(s) if c == " "]
                for st in starts:
                    frag = s[st:st + n].strip()
                    # scale down for dense scripts (one kana/kanji carries ~2-3 Latin chars of information)
                    if lang in NO_SPACE:
                        frag = s[st:st + max(2, n // 2)]
                    if len(frag) >= min(n, 3) - 1:
                        x.append(frag); y.append(L_IDX[lang])
        sets.append((x, np.array(y))); out["n_per_bucket"].append(len(x))
    for m in ("nb", "lr", "fasttext"):
        accs, latin = [], []
        for x, y in sets:
            P, _ = classify(x, m, S=S)
            ok = P.argmax(1) == y
            lat = np.isin(y, [L_IDX[l] for l in SCRIPT_LANGS["Latin"]])
            accs.append(round(float(ok.mean()), 4)); latin.append(round(float(ok[lat].mean()), 4))
        out["series"].append({"model": m, "name": MODEL_NAMES[m], "accuracy": accs, "latin_only_accuracy": latin})
    return out


# ---- class imbalance -----------------------------------------------------------------
class ImbalanceIn(BaseModel):
    minority_lang: str = "uk"
    minority_samples: int = Field(8, ge=2, le=200)
    model: str = "nb"


@lru_cache(maxsize=64)
def _imbalance(lang: str, n_min: int, model: str):
    S = state()
    rng = np.random.default_rng(3)
    y = S.y_train; li = L_IDX[lang]
    keep = np.r_[np.where(y != li)[0], rng.permutation(np.where(y == li)[0])[:n_min]]
    X, yy = S.X_train[keep], y[keep]
    tx, ty = _xy(S.test)
    sib = [c for c in SCRIPT_LANGS[next(s for s, ls in SCRIPT_LANGS.items() if lang in ls)] if c != lang]
    n_maj = int(np.median(np.bincount(yy, minlength=len(LANGS))))
    res = []
    for strat in ("none", "class_weight", "oversample"):
        Xs, ys, kw = X, yy, {}
        if strat == "oversample":                      # duplicate minority rows up to the median class size
            extra = rng.choice(np.where(yy == li)[0], max(0, n_maj - n_min))
            Xs = sparse.vstack([X, X[extra]]).tocsr(); ys = np.r_[yy, yy[extra]]
        if model == "nb":
            # NB: class weighting == a uniform class prior instead of the empirical (skewed) one
            clf = MultinomialNB(alpha=0.05, fit_prior=(strat != "class_weight")).fit(Xs, ys)
        else:
            clf = LogisticRegression(C=20.0, max_iter=40, class_weight="balanced" if strat == "class_weight" else None).fit(Xs, ys)
        P, _ = classify(tx, model, True, True, S, clf=clf)
        mt = metrics(ty, P)
        per = {p["lang"]: p for p in mt["per_language"]}
        row = {"strategy": strat, "accuracy": mt["accuracy"], "macro_f1": mt["macro_f1"],
               f"{lang}_recall": per[lang]["recall"], f"{lang}_precision": per[lang]["precision"], f"{lang}_f1": per[lang]["f1"]}
        for s in sib[:1]:
            row[f"{s}_recall"] = per[s]["recall"]; row[f"{s}_precision"] = per[s]["precision"]
        res.append(row)
    return {"minority_lang": lang, "minority_samples": n_min, "majority_samples": n_maj, "ratio": round(n_maj / n_min, 1),
            "model": model, "results": res,
            "note": f"Accuracy barely moves ({lang} is 1/{len(LANGS)} of the test set) while {lang} recall collapses - "
                    "report macro-F1 and per-language recall, never just accuracy."}


@router.post("/imbalance")
def imbalance(q: ImbalanceIn):
    """Low-resource language: shrink its training data, compare none / class weights / oversampling."""
    if q.minority_lang not in LANGS or q.model not in ("nb", "lr"):
        raise HTTPException(400, "bad minority_lang or model (nb|lr)")
    if len(next(ls for ls in SCRIPT_LANGS.values() if q.minority_lang in ls)) < 2:
        raise HTTPException(400, "pick a language that shares its script with another (the script rule solves the rest)")
    return _imbalance(q.minority_lang, q.minority_samples, q.model)


# ---- pitfalls --------------------------------------------------------------------------
class PitfallIn(BaseModel):
    kind: str = "leakage"       # leakage | domain_shift | noise
    fix: bool = False


def _fit_nb(rows):
    x, y = _xy(rows)
    vec = _tfidf(); X = vec.fit_transform(_pad(x))
    return vec, MultinomialNB(alpha=0.05).fit(X, y)


@lru_cache(maxsize=8)
def _pitfall(kind: str):
    S = state()
    rng = np.random.default_rng(5)
    if kind == "leakage":
        # Sentences A (train ids), B (4 held-out ids) and F (4 'fresh' ids standing in for production).
        B, F = TEST_SENT[:4], TEST_SENT[4:]
        pool, fresh = [], []
        for lang in LANGS:
            pool += _synth(lang, TRAIN_SENT + B, 220, rng, min_units=2)
            fresh += _synth(lang, F, 60, rng, min_units=2)
        fx, fy = _xy(fresh)
        # WRONG: random row split -> windows of the same sentence land on both sides
        perm = rng.permutation(len(pool)); cut = int(0.8 * len(pool))
        tr, te = [pool[i] for i in perm[:cut]], [pool[i] for i in perm[cut:]]
        vec, clf = _fit_nb(tr)
        tex, tey = _xy(te)
        tr_texts = {t for t, _, _ in tr}; tr_groups = {(l, s) for _, l, s in tr}
        wrong = {"split": "random rows", "reported_offline_acc": metrics(tey, classify(tex, "nb", S=S, vec=vec, clf=clf)[0])["accuracy"],
                 "fresh_sentences_acc": metrics(fy, classify(fx, "nb", S=S, vec=vec, clf=clf)[0])["accuracy"],
                 "test_rows_exact_dup_in_train": round(float(np.mean([t in tr_texts for t in tex])), 3),
                 "test_rows_same_source_sentence_in_train": round(float(np.mean([(l, s) in tr_groups for _, l, s in te])), 3)}
        # RIGHT: group split by source sentence + exact-dedup against train
        tr = [r for r in pool if r[2] not in B]; te = [r for r in pool if r[2] in B]
        vec, clf = _fit_nb(tr)
        tr_texts = {t for t, _, _ in tr}
        te = [r for r in te if r[0] not in tr_texts]
        tex, tey = _xy(te)
        right = {"split": "grouped by source sentence + dedup", "reported_offline_acc": metrics(tey, classify(tex, "nb", S=S, vec=vec, clf=clf)[0])["accuracy"],
                 "fresh_sentences_acc": metrics(fy, classify(fx, "nb", S=S, vec=vec, clf=clf)[0])["accuracy"],
                 "test_rows_exact_dup_in_train": 0.0, "test_rows_same_source_sentence_in_train": 0.0}
        for d in (wrong, right):
            d["optimism_gap"] = round(d["reported_offline_acc"] - d["fresh_sentences_acc"], 4)
        return {"kind": kind, "wrong": wrong, "right": right,
                "explanation": "Posts get quoted, retweeted and templated; a random split puts near-duplicates of the same source on both sides, "
                               "so the offline number measures memorisation. Split by source (author / thread / near-dup cluster) and dedup: "
                               "the reported number drops but now predicts accuracy on genuinely new text."}
    if kind == "domain_shift":
        tx, ty = _xy(S.test); sx, sy = _xy(S.social)
        out = {}
        for name, formal_only in (("wrong", True), ("right", False)):
            rows = make_train(formal_only=formal_only)
            vec, clf = _fit_nb(rows)
            ms = metrics(sy, classify(sx, "nb", S=S, vec=vec, clf=clf)[0])
            per = sorted(ms["per_language"], key=lambda p: p["recall"])[:3]
            out[name] = {"training_data": "formal sentences + lexicon only" if formal_only else "formal + style augmentation (accent stripping, elongation, lowercase) + ~15 labelled social samples / language",
                         "train_rows": len(rows),
                         "formal_test_acc": metrics(ty, classify(tx, "nb", S=S, vec=vec, clf=clf)[0])["accuracy"],
                         "social_test_acc": ms["accuracy"], "social_macro_f1": ms["macro_f1"],
                         "worst_social_languages": ", ".join(f"{p['lang']} (recall {p['recall']})" for p in per)}
            out[name]["domain_gap"] = round(out[name]["formal_test_acc"] - out[name]["social_test_acc"], 4)
        return {"kind": kind, **out,
                "explanation": "Wikipedia/news-style training text has accents, full words and punctuation; posts have 'q', 'vc', 'jsuis', 'nn', dropped accents "
                               "and dialect. The offline (formal) metric looks fine in both cases - only an in-domain test set reveals the gap. "
                               "Fix with augmentation that mimics the target style plus a small in-domain labelled set; keep a held-out social slice as the release gate."}
    if kind == "noise":
        tx, ty = _xy(S.test)
        idx = rng.permutation(len(tx))[:480]
        base = [tx[i] for i in idx]; y = ty[idx]
        rows = []
        for label, kinds in (("clean text", ()), ("+ emoji", ("emoji",)), ("+ @mention", ("mention",)), ("+ #hashtags", ("hashtag",)),
                             ("+ URL", ("url",)), ("all of the above", ("url", "mention", "hashtag", "emoji"))):
            r = np.random.default_rng(9)
            noisy = [add_noise(t, r, kinds) for t in base] if kinds else base
            P_raw, st_raw = classify(noisy, "lr", True, False, S)
            P_fix, _ = classify(noisy, "lr", True, True, S)
            nl = ~np.isin(y, [L_IDX[l] for l in SCRIPT_LANGS["Latin"]])
            rows.append({"noise": label, "raw_acc": metrics(y, P_raw)["accuracy"], "preprocessed_acc": metrics(y, P_fix)["accuracy"],
                         "raw_acc_non_latin": round(float((P_raw.argmax(1) == y)[nl].mean()), 4),
                         "preprocessed_acc_non_latin": round(float((P_fix.argmax(1) == y)[nl].mean()), 4)})
        ex = add_noise("شكرا يا صديقي", np.random.default_rng(1))
        return {"kind": kind, "rows": rows,
                "wrong": {"pipeline": "raw text -> script rule -> model", "accuracy_all_noise": rows[-1]["raw_acc"],
                          "accuracy_non_latin_all_noise": rows[-1]["raw_acc_non_latin"], "accuracy_clean": rows[0]["raw_acc"]},
                "right": {"pipeline": "strip URLs/@/#/emoji/digits, squeeze elongation -> script rule -> model", "accuracy_all_noise": rows[-1]["preprocessed_acc"],
                          "accuracy_non_latin_all_noise": rows[-1]["preprocessed_acc_non_latin"], "accuracy_clean": rows[0]["preprocessed_acc"]},
                "example": {"raw": ex, "cleaned": clean(ex)[0], "raw_script": script_profile(ex)["shares"], "cleaned_script": script_profile(clean(ex)[0])["shares"]},
                "explanation": "URLs, handles and hashtags are Latin-script and English-looking whatever the post language. Raw, they flip the script vote of short "
                               "Arabic/Hindi/Japanese/Cyrillic posts to Latin (stage 1 fails) and drag Latin posts towards English (stage 2 fails). "
                               "Emoji alone barely matter: they are not in the n-gram vocabulary. Preprocessing must be IDENTICAL in training and serving."}
    raise HTTPException(400, "kind must be leakage | domain_shift | noise")


@router.post("/pitfall")
def pitfall(q: PitfallIn):
    """Three classic LID pitfalls; `fix` toggles which pipeline is 'shown' (both are returned for comparison)."""
    r = dict(_pitfall(q.kind))
    r["fix"] = q.fix
    r["shown"] = r["right" if q.fix else "wrong"]
    return r
