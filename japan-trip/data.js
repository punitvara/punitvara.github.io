// ============================================================
// EDIT THIS FILE. dashboard.html reads everything from here.
// status values: "idea" | "shortlisted" | "booked" | "confirmed"
// ============================================================
const TRIP = {
  title: "Japan Trip – April",
  travelers: "2 adults + 1 senior (mom) + 1 infant (<1 yr)",
  passport: "Indian",
  startDate: "2027-04-02",    // depart BLR
  endDate: "2027-04-13",      // depart KIX (fixed by user 2026-10-20: late-night flight, Kyoto Airbnb checkout same day)
  notes: "First international trip. Infant: needs own passport + visa. Keep days light, hotel near stations."
};

// Timeline: each block is one bar. days = number of days.
const TIMELINE = [
  { label: "Fly BLR → Tokyo",   days: 1, type: "travel" },
  { label: "Tokyo (5 nights, 3–8 Apr)",  days: 5, type: "city" },
  { label: "Kyoto (5 nights, 8–13 Apr)",  days: 5, type: "city" },
  { label: "Fly KIX → BLR",     days: 1, type: "travel" }
];

// Map points. kind: airport | station | sight | hotel
const PLACES = [
  { name: "Haneda Airport (HND)",  city: "Tokyo", lat: 35.5494, lng: 139.7798, kind: "airport", day: 1 },
  { name: "Ueno Park",             city: "Tokyo", lat: 35.7156, lng: 139.7745, kind: "sight",   day: 1 },
  { name: "Senso-ji, Asakusa",     city: "Tokyo", lat: 35.7148, lng: 139.7967, kind: "sight",   day: 2 },
  { name: "Shinjuku Gyoen",        city: "Tokyo", lat: 35.6852, lng: 139.7100, kind: "sight",   day: 2 },
  { name: "Shibuya Crossing",      city: "Tokyo", lat: 35.6595, lng: 139.7005, kind: "sight",   day: 2 },
  { name: "Chidorigafuchi (blossom, flex)", city: "Tokyo", lat: 35.6918, lng: 139.7448, kind: "sight", day: 3 },
  { name: "Shinjuku Station (Tokyo stay area)", city: "Tokyo", lat: 35.6896, lng: 139.7006, kind: "hotel", day: 0 },
  { name: "Tokyo Station",         city: "Tokyo", lat: 35.6812, lng: 139.7671, kind: "station", day: 4 },
  { name: "Kyoto Station",         city: "Kyoto", lat: 34.9858, lng: 135.7588, kind: "station", day: 4 },
  { name: "Fushimi Inari",         city: "Kyoto", lat: 34.9671, lng: 135.7727, kind: "sight",   day: 5 },
  { name: "Kiyomizu-dera",         city: "Kyoto", lat: 34.9949, lng: 135.7850, kind: "sight",   day: 5 },
  { name: "Gion",                  city: "Kyoto", lat: 35.0037, lng: 135.7756, kind: "sight",   day: 5 },
  { name: "Arashiyama Bamboo",     city: "Kyoto", lat: 35.0170, lng: 135.6713, kind: "sight",   day: 6 },
  { name: "Kinkaku-ji",            city: "Kyoto", lat: 35.0394, lng: 135.7292, kind: "sight",   day: 7 },
  { name: "Nishiki Market",        city: "Kyoto", lat: 35.0050, lng: 135.7647, kind: "sight",   day: 7 },
  { name: "Kansai Airport (KIX)",  city: "Osaka", lat: 34.4320, lng: 135.2304, kind: "airport", day: 8 }
];

// Day-by-day plan (draft). Edit freely.
const ITINERARY = [
  { day: "Apr 2 (Fri)",  city: "Bengaluru", morning: "Depart BLR 11:35 (SQ509)", afternoon: "Singapore layover 3h50 (SIN T3→T2)", evening: "SQ636 departs SIN 22:50, overnight flight" },
  { day: "Apr 3 (Sat)",  city: "Tokyo", morning: "Land HND T3 06:45, immigration, hotel luggage drop", afternoon: "Ueno Park (blossom, flat, stroller-friendly)", evening: "Early dinner, rest" },
  { day: "Apr 4 (Sun)",  city: "Tokyo", morning: "Senso-ji, Asakusa", afternoon: "Shinjuku Gyoen (check entry & stroller rules)", evening: "Shibuya Crossing, back by 7pm" },
  { day: "Apr 5 (Mon)",  city: "Tokyo", morning: "Flex: Chidorigafuchi / Meguro River if in bloom", afternoon: "Rest / shopping", evening: "Pack" },
  { day: "Apr 6 (Tue)",  city: "Tokyo", morning: "Meiji Shrine (forest walk, flat) / Omotesando", afternoon: "Rest at hotel", evening: "Early dinner nearby" },
  { day: "Apr 7 (Wed)",  city: "Tokyo", morning: "Flex: Imperial Palace East Gardens / Ginza", afternoon: "Rest / shopping", evening: "Pack, send luggage ahead (takkyubin)" },
  { day: "Apr 8 (Thu)",  city: "Tokyo → Kyoto", morning: "Check out Tokyo hotel, Tokyo Station", afternoon: "Tokaido Shinkansen Nozomi ~2h15, check in Kyoto hotel (≈3h door to door)", evening: "Kyoto Station area dinner" },
  { day: "Apr 9 (Fri)",  city: "Kyoto", morning: "Fushimi Inari (go early)", afternoon: "Kiyomizu-dera, Higashiyama", evening: "Gion walk, early night" },
  { day: "Apr 10 (Sat)", city: "Kyoto", morning: "Arashiyama bamboo grove", afternoon: "Rest at hotel", evening: "Flex" },
  { day: "Apr 11 (Sun)", city: "Kyoto", morning: "Kinkaku-ji / Philosopher's Path", afternoon: "Nishiki Market, Nijo Castle", evening: "Buffer" },
  { day: "Apr 12 (Mon)", city: "Kyoto", morning: "Flex: Nara day trip (deer park, 45 min by JR) or Osaka half-day", afternoon: "Pack, souvenirs", evening: "Early night" },
  { day: "Apr 13 (Tue)", city: "Kyoto → KIX", morning: "Check out of Airbnb, store luggage", afternoon: "Free time near Kyoto Station", evening: "Haruka to KIX ~17:00–18:00, SQ flight departs 23:25 (arrive BLR 10:40 on 14 Apr)" }
];

// Candidates / options for comparison (02-options)
const OPTIONS = [
  { category: "Flight",  name: "JAL nonstop BLR → Tokyo (return from KIX needs open-jaw check)", price: null, notes: "Confidence LOW: JAL showed ~₹88k RT for mid-2026 dates only (direct fetch 403). Get live April 2027 quote.", status: "idea" },
  { category: "Flight",  name: "IndiGo nonstop BLR → Tokyo NRT", price: null, notes: "Confidence LOW: named nonstop by Skyscanner/Kupi. Verify April 2027 operation, bassinet.", status: "idea" },
  { category: "Flight",  name: "Singapore Airlines BLR→SIN→HND/NRT, KIX→SIN→BLR", price: 40000, notes: "Confidence MEDIUM: Goibibo ~₹39,836 one-way (Nov date). Google Flights RT $720–1,450. Ask for ≥2.5h layover.", status: "idea" },
  { category: "Hotel",   name: "Tokyo: Ueno or Shinjuku family room (5 nights, 3–8 Apr)", price: null, notes: "Confidence LOW: no live rates. Candidates: Keio Plaza, Mimaru apartments.", status: "idea" },
  { category: "Hotel",   name: "Kyoto: Mimaru Kyoto Station / Granvia (5 nights, 8–13 Apr)", price: null, notes: "Confidence LOW: Mimaru ¥35–65k/night (undated).", status: "idea" },
  { category: "Transit", name: "Tokyo → Kyoto Nozomi reserved", price: 25000, notes: "3 adults ¥42.5–43.1k ≈ ₹24–27k; infant free on lap (verify). ¥14,170–14,370 per adult (japantrain.net). Book on SmartEX.", status: "idea" },
  { category: "Transit", name: "Kyoto → KIX Haruka express", price: 6000, notes: "3 adults ¥8.5–10.5k ≈ ₹5–6k; infant free on lap (verify). Buy at Kyoto Station JR office.", status: "idea" }
];

// Confirmed bookings only (04-bookings). Put files in that folder, list here.
const BOOKINGS = [
  // { type: "Hotel", name: "...", dates: "...", ref: "ABC123", file: "04-bookings/hotel-tokyo.pdf", status: "confirmed" }
];

// Things with a deadline. date: "YYYY-MM-DD" or null if unknown.
const TODO = [
  { task: "Check passports valid (all 4, infant needs own)", date: null, done: false },
  { task: "Apply Japan tourist visa (all 4 incl. infant)",    date: null, done: false },
  { task: "Fix travel dates (check cherry blossom forecast)", date: null, done: false },
  { task: "Book flights",                                     date: null, done: false },
  { task: "Book hotels (free cancellation)",                  date: null, done: false },
  { task: "Travel insurance",                                 date: null, done: false },
  { task: "Book Shinkansen seats (opens ~1 month before)",    date: null, done: false },
  { task: "Pocket WiFi / eSIM",                               date: null, done: false }
];

// Budget in INR. Fill planned/actual; null = unknown.
const BUDGET = [
  { cat: "Flights",       planned: null, actual: null },
  { cat: "Hotels",        planned: null, actual: null },
  { cat: "Local transit", planned: null, actual: null },
  { cat: "Food",          planned: null, actual: null },
  { cat: "Activities",    planned: null, actual: null },
  { cat: "Visa/Insurance",planned: null, actual: null },
  { cat: "Buffer",        planned: null, actual: null }
];

// Decisions & things to remember. level: "decision" | "tip" | "warning"
const NOTES = [
  { level: "decision", text: "Stay area: Tokyo = 3–8 Apr (5 nights), Kyoto = 8–13 Apr (5 nights, near Kyoto Station). Check-in/out dates fixed by user." },
  { level: "tip",      text: "Shinjuku → Kyoto Station by Shinkansen ≈ 3 h door to door: Shinjuku→Tokyo Station (JR Chuo Rapid) ~15–20 min + ~15–20 min buffer + Nozomi ~2h15–20. Shinkansen does NOT stop at Shinjuku." },
  { level: "tip",      text: "Kyoto Station → Kansai Airport (KIX) by JR Haruka limited express ≈ 75–80 min, direct, no change. Plan to leave Kyoto ≥ 4 h before an international departure (≈ 2.5–3 h before flight at airport + 80 min train + buffer)." },
  { level: "tip",      text: "Nozomi (fastest) is not covered by JR Pass. Hikari ~2h40, Kodama ~3h50." },
  { level: "tip",      text: "Journey timing: 8 Apr hotel→hotel ≈ 3h05–3h40. 13 Apr hotel→KIX check-in ≈ 1h55–2h25, plus 2.5–3 h at airport, so leave hotel ≈ 4h30 before the flight. See trains.html." },
  { level: "tip",      text: "Exact trains: see trains.html. 8 Apr = Chuo Rapid (Shinjuku→Tokyo) then Tokaido Shinkansen Nozomi to Kyoto. 13 Apr = JR Haruka from Kyoto Platform 30 direct to KIX." },
  { level: "warning",  text: "Large suitcases on Shinkansen may need an oversize-baggage reservation; consider takkyubin luggage forwarding Shinjuku hotel → Kyoto hotel." },
  { level: "warning",  text: "Verify before booking: infant fare for Shinkansen and Haruka (JR Central / JR West), exact Haruka timetable for 13 April." }
];

// Flight options tracker. leg: "out" (BLR→Tokyo), "ret" (KIX→BLR) or "rt" (one ticket covering both).
// price = TOTAL INR for 3 adults + 1 infant. null = not priced yet. Pick one per leg on the dashboard.
const FLIGHTS = [
  { id: "sq-out-1", leg: "out", airline: "Singapore Airlines", route: "BLR 11:35 → SIN (3h50 layover) → HND 06:45+1", duration: "15h 40m", price: 157689, was: 172041, source: "Amazon, screenshot 2026-10-19", notes: "Confidence HIGH for quote; date, baggage, refund rules unverified. Compare on singaporeair.com." },
  { id: "sq-ret-1", leg: "ret", airline: "Singapore Airlines", route: "13 Apr: KIX 23:25 → SIN (3h55 layover) → BLR 10:40+1", duration: "14h 45m", price: 126034, was: 138719, source: "Amazon, screenshot 2026-10-19", notes: "Departs 13 Apr (3 days later than the 10 Apr plan). Pax assumed 3A+1I like the outbound; confirm. Late flight: use the day for checkout and Haruka at leisure. Compare on singaporeair.com." },
  { id: "sq-out-scapia", leg: "out", airline: "Singapore Airlines (Scapia)", route: "2 Apr BLR → SIN (3h50 layover) → SQ636 SIN 22:50 → HND 06:45+1 (T3)", duration: "15h 40m", price: 158341, was: 164541, source: "Scapia app, screenshot 2026-10-20", notes: "Same flights as Amazon, ₹652 dearer before rewards (Flat ₹6,000 off already applied). Standard economy, 30 kg bag, cancellation from ₹9,964, date change from ₹1,995. Coins back not shown for this leg (return earns about 0.84% in coin value, so outbound est. about ₹1,330). 4 travellers." },
  { id: "sq-ret-scapia-2325", leg: "ret", airline: "Singapore Airlines (Scapia)", route: "13 Apr: KIX 23:25 → SIN (3h55 layover) → BLR 10:40+1", duration: "14h 45m", price: 126001, was: 132201, source: "Scapia app, screenshot 2026-10-20", notes: "₹33 cheaper than Amazon before rewards; shows 5,280 coins back (5 coins = ₹1, so about ₹1,056; effective about ₹1,24,945; coins only usable on Scapia). Outbound coins not seen. 4 travellers." },
];
