#!/usr/bin/env python3
"""Generate a large factual QA dataset (1000+ pairs) with verifiable answers.

Categories:
  - Math: arithmetic with random operands (trivially verifiable)
  - Geography: country capitals and approximate populations
  - Science: element symbols, atomic numbers from the periodic table
  - Dates: well-known historical events with years
  - General: miscellaneous countable/factual trivia

Each entry is {"question": str, "expected_answer_substring": str, "category": str}.

The dataset is intentionally large so that EBM activation-steering experiments
can train on 1000+ examples, avoiding the overfitting risk seen with 25-93
example datasets. Every answer is a short substring that MUST appear in a
correct free-text response (e.g. "Paris" for "What is the capital of France?").
"""

import json
import random
import pathlib
from typing import List, Dict

# ---------------------------------------------------------------------------
# 1. MATH — random arithmetic (400 pairs)
# ---------------------------------------------------------------------------

def _generate_math_pairs(n: int = 400, seed: int = 42) -> List[Dict[str, str]]:
    """Produce n arithmetic QA pairs.

    We mix multiplication, addition, and subtraction so the model can't just
    pattern-match on "What is A * B?" phrasing.
    """
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    templates = [
        # (question_template, op_func, op_symbol)
        ("What is {a} * {b}?", lambda a, b: a * b, "*"),
        ("What is {a} + {b}?", lambda a, b: a + b, "+"),
        ("What is {a} - {b}?", lambda a, b: a - b, "-"),
        ("What is {a} times {b}?", lambda a, b: a * b, "times"),
        ("Calculate {a} multiplied by {b}.", lambda a, b: a * b, "multiplied"),
        ("What do you get when you add {a} and {b}?", lambda a, b: a + b, "add"),
    ]
    for _ in range(n):
        tmpl, op, _ = rng.choice(templates)
        a = rng.randint(2, 999)
        b = rng.randint(2, 999)
        question = tmpl.format(a=a, b=b)
        answer = str(op(a, b))
        pairs.append({
            "question": question,
            "expected_answer_substring": answer,
            "category": "math",
        })
    return pairs


# ---------------------------------------------------------------------------
# 2. GEOGRAPHY — capitals & approximate populations (200+ countries)
# ---------------------------------------------------------------------------

# Hardcoded list of (country, capital, approximate_population_description).
# Population figures are rounded/approximate and used as substring targets.
COUNTRIES = [
    ("Afghanistan", "Kabul", "40 million"),
    ("Albania", "Tirana", "2.8 million"),
    ("Algeria", "Algiers", "45 million"),
    ("Andorra", "Andorra la Vella", "80,000"),
    ("Angola", "Luanda", "36 million"),
    ("Antigua and Barbuda", "St. John's", "100,000"),
    ("Argentina", "Buenos Aires", "46 million"),
    ("Armenia", "Yerevan", "3 million"),
    ("Australia", "Canberra", "26 million"),
    ("Austria", "Vienna", "9 million"),
    ("Azerbaijan", "Baku", "10 million"),
    ("Bahamas", "Nassau", "400,000"),
    ("Bahrain", "Manama", "1.5 million"),
    ("Bangladesh", "Dhaka", "170 million"),
    ("Barbados", "Bridgetown", "280,000"),
    ("Belarus", "Minsk", "9.4 million"),
    ("Belgium", "Brussels", "11.5 million"),
    ("Belize", "Belmopan", "400,000"),
    ("Benin", "Porto-Novo", "13 million"),
    ("Bhutan", "Thimphu", "780,000"),
    ("Bolivia", "Sucre", "12 million"),
    ("Bosnia and Herzegovina", "Sarajevo", "3.2 million"),
    ("Botswana", "Gaborone", "2.4 million"),
    ("Brazil", "Brasilia", "215 million"),
    ("Brunei", "Bandar Seri Begawan", "450,000"),
    ("Bulgaria", "Sofia", "6.5 million"),
    ("Burkina Faso", "Ouagadougou", "22 million"),
    ("Burundi", "Gitega", "13 million"),
    ("Cabo Verde", "Praia", "590,000"),
    ("Cambodia", "Phnom Penh", "17 million"),
    ("Cameroon", "Yaounde", "28 million"),
    ("Canada", "Ottawa", "40 million"),
    ("Central African Republic", "Bangui", "5 million"),
    ("Chad", "N'Djamena", "17 million"),
    ("Chile", "Santiago", "19 million"),
    ("China", "Beijing", "1.4 billion"),
    ("Colombia", "Bogota", "52 million"),
    ("Comoros", "Moroni", "900,000"),
    ("Congo", "Brazzaville", "6 million"),
    ("Costa Rica", "San Jose", "5 million"),
    ("Croatia", "Zagreb", "3.9 million"),
    ("Cuba", "Havana", "11 million"),
    ("Cyprus", "Nicosia", "1.2 million"),
    ("Czech Republic", "Prague", "10.5 million"),
    ("Democratic Republic of the Congo", "Kinshasa", "100 million"),
    ("Denmark", "Copenhagen", "5.9 million"),
    ("Djibouti", "Djibouti", "1 million"),
    ("Dominica", "Roseau", "72,000"),
    ("Dominican Republic", "Santo Domingo", "11 million"),
    ("East Timor", "Dili", "1.3 million"),
    ("Ecuador", "Quito", "18 million"),
    ("Egypt", "Cairo", "105 million"),
    ("El Salvador", "San Salvador", "6.3 million"),
    ("Equatorial Guinea", "Malabo", "1.7 million"),
    ("Eritrea", "Asmara", "3.6 million"),
    ("Estonia", "Tallinn", "1.3 million"),
    ("Eswatini", "Mbabane", "1.2 million"),
    ("Ethiopia", "Addis Ababa", "120 million"),
    ("Fiji", "Suva", "930,000"),
    ("Finland", "Helsinki", "5.5 million"),
    ("France", "Paris", "68 million"),
    ("Gabon", "Libreville", "2.3 million"),
    ("Gambia", "Banjul", "2.5 million"),
    ("Georgia", "Tbilisi", "3.7 million"),
    ("Germany", "Berlin", "84 million"),
    ("Ghana", "Accra", "33 million"),
    ("Greece", "Athens", "10.4 million"),
    ("Grenada", "St. George's", "125,000"),
    ("Guatemala", "Guatemala City", "17 million"),
    ("Guinea", "Conakry", "14 million"),
    ("Guinea-Bissau", "Bissau", "2 million"),
    ("Guyana", "Georgetown", "800,000"),
    ("Haiti", "Port-au-Prince", "11.5 million"),
    ("Honduras", "Tegucigalpa", "10 million"),
    ("Hungary", "Budapest", "9.7 million"),
    ("Iceland", "Reykjavik", "380,000"),
    ("India", "New Delhi", "1.4 billion"),
    ("Indonesia", "Jakarta", "275 million"),
    ("Iran", "Tehran", "87 million"),
    ("Iraq", "Baghdad", "43 million"),
    ("Ireland", "Dublin", "5 million"),
    ("Israel", "Jerusalem", "9.5 million"),
    ("Italy", "Rome", "59 million"),
    ("Ivory Coast", "Yamoussoukro", "28 million"),
    ("Jamaica", "Kingston", "2.8 million"),
    ("Japan", "Tokyo", "125 million"),
    ("Jordan", "Amman", "11 million"),
    ("Kazakhstan", "Astana", "19 million"),
    ("Kenya", "Nairobi", "55 million"),
    ("Kiribati", "Tarawa", "120,000"),
    ("Kuwait", "Kuwait City", "4.3 million"),
    ("Kyrgyzstan", "Bishkek", "7 million"),
    ("Laos", "Vientiane", "7.5 million"),
    ("Latvia", "Riga", "1.8 million"),
    ("Lebanon", "Beirut", "5.5 million"),
    ("Lesotho", "Maseru", "2.3 million"),
    ("Liberia", "Monrovia", "5.3 million"),
    ("Libya", "Tripoli", "7 million"),
    ("Liechtenstein", "Vaduz", "39,000"),
    ("Lithuania", "Vilnius", "2.8 million"),
    ("Luxembourg", "Luxembourg City", "650,000"),
    ("Madagascar", "Antananarivo", "29 million"),
    ("Malawi", "Lilongwe", "20 million"),
    ("Malaysia", "Kuala Lumpur", "33 million"),
    ("Maldives", "Male", "520,000"),
    ("Mali", "Bamako", "22 million"),
    ("Malta", "Valletta", "530,000"),
    ("Marshall Islands", "Majuro", "42,000"),
    ("Mauritania", "Nouakchott", "4.8 million"),
    ("Mauritius", "Port Louis", "1.3 million"),
    ("Mexico", "Mexico City", "130 million"),
    ("Micronesia", "Palikir", "115,000"),
    ("Moldova", "Chisinau", "2.6 million"),
    ("Monaco", "Monaco", "39,000"),
    ("Mongolia", "Ulaanbaatar", "3.4 million"),
    ("Montenegro", "Podgorica", "620,000"),
    ("Morocco", "Rabat", "37 million"),
    ("Mozambique", "Maputo", "33 million"),
    ("Myanmar", "Naypyidaw", "54 million"),
    ("Namibia", "Windhoek", "2.5 million"),
    ("Nauru", "Yaren", "12,000"),
    ("Nepal", "Kathmandu", "30 million"),
    ("Netherlands", "Amsterdam", "17.5 million"),
    ("New Zealand", "Wellington", "5 million"),
    ("Nicaragua", "Managua", "7 million"),
    ("Niger", "Niamey", "26 million"),
    ("Nigeria", "Abuja", "220 million"),
    ("North Korea", "Pyongyang", "26 million"),
    ("North Macedonia", "Skopje", "1.8 million"),
    ("Norway", "Oslo", "5.4 million"),
    ("Oman", "Muscat", "4.5 million"),
    ("Pakistan", "Islamabad", "230 million"),
    ("Palau", "Ngerulmud", "18,000"),
    ("Palestine", "Ramallah", "5.4 million"),
    ("Panama", "Panama City", "4.4 million"),
    ("Papua New Guinea", "Port Moresby", "10 million"),
    ("Paraguay", "Asuncion", "7 million"),
    ("Peru", "Lima", "34 million"),
    ("Philippines", "Manila", "115 million"),
    ("Poland", "Warsaw", "38 million"),
    ("Portugal", "Lisbon", "10 million"),
    ("Qatar", "Doha", "2.9 million"),
    ("Romania", "Bucharest", "19 million"),
    ("Russia", "Moscow", "144 million"),
    ("Rwanda", "Kigali", "14 million"),
    ("Saint Kitts and Nevis", "Basseterre", "47,000"),
    ("Saint Lucia", "Castries", "180,000"),
    ("Saint Vincent and the Grenadines", "Kingstown", "110,000"),
    ("Samoa", "Apia", "220,000"),
    ("San Marino", "San Marino", "34,000"),
    ("Sao Tome and Principe", "Sao Tome", "230,000"),
    ("Saudi Arabia", "Riyadh", "36 million"),
    ("Senegal", "Dakar", "17 million"),
    ("Serbia", "Belgrade", "6.7 million"),
    ("Seychelles", "Victoria", "100,000"),
    ("Sierra Leone", "Freetown", "8.5 million"),
    ("Singapore", "Singapore", "5.5 million"),
    ("Slovakia", "Bratislava", "5.4 million"),
    ("Slovenia", "Ljubljana", "2.1 million"),
    ("Solomon Islands", "Honiara", "720,000"),
    ("Somalia", "Mogadishu", "18 million"),
    ("South Africa", "Pretoria", "60 million"),
    ("South Korea", "Seoul", "52 million"),
    ("South Sudan", "Juba", "11 million"),
    ("Spain", "Madrid", "47 million"),
    ("Sri Lanka", "Sri Jayawardenepura Kotte", "22 million"),
    ("Sudan", "Khartoum", "46 million"),
    ("Suriname", "Paramaribo", "620,000"),
    ("Sweden", "Stockholm", "10.5 million"),
    ("Switzerland", "Bern", "8.8 million"),
    ("Syria", "Damascus", "22 million"),
    ("Taiwan", "Taipei", "23 million"),
    ("Tajikistan", "Dushanbe", "10 million"),
    ("Tanzania", "Dodoma", "65 million"),
    ("Thailand", "Bangkok", "72 million"),
    ("Togo", "Lome", "8.8 million"),
    ("Tonga", "Nuku'alofa", "100,000"),
    ("Trinidad and Tobago", "Port of Spain", "1.4 million"),
    ("Tunisia", "Tunis", "12 million"),
    ("Turkey", "Ankara", "85 million"),
    ("Turkmenistan", "Ashgabat", "6.3 million"),
    ("Tuvalu", "Funafuti", "11,000"),
    ("Uganda", "Kampala", "48 million"),
    ("Ukraine", "Kyiv", "37 million"),
    ("United Arab Emirates", "Abu Dhabi", "10 million"),
    ("United Kingdom", "London", "67 million"),
    ("United States", "Washington, D.C.", "333 million"),
    ("Uruguay", "Montevideo", "3.4 million"),
    ("Uzbekistan", "Tashkent", "35 million"),
    ("Vanuatu", "Port Vila", "320,000"),
    ("Vatican City", "Vatican City", "800"),
    ("Venezuela", "Caracas", "28 million"),
    ("Vietnam", "Hanoi", "100 million"),
    ("Yemen", "Sanaa", "33 million"),
    ("Zambia", "Lusaka", "20 million"),
    ("Zimbabwe", "Harare", "16 million"),
]


def _generate_geography_pairs() -> List[Dict[str, str]]:
    """Two questions per country: capital and population. ~400 pairs."""
    pairs: List[Dict[str, str]] = []
    for country, capital, pop in COUNTRIES:
        pairs.append({
            "question": f"What is the capital of {country}?",
            "expected_answer_substring": capital,
            "category": "geography",
        })
        pairs.append({
            "question": f"What is the approximate population of {country}?",
            "expected_answer_substring": pop,
            "category": "geography",
        })
    return pairs


# ---------------------------------------------------------------------------
# 3. SCIENCE — periodic table (first 118 elements)
# ---------------------------------------------------------------------------

# (atomic_number, symbol, name)
ELEMENTS = [
    (1, "H", "Hydrogen"), (2, "He", "Helium"), (3, "Li", "Lithium"),
    (4, "Be", "Beryllium"), (5, "B", "Boron"), (6, "C", "Carbon"),
    (7, "N", "Nitrogen"), (8, "O", "Oxygen"), (9, "F", "Fluorine"),
    (10, "Ne", "Neon"), (11, "Na", "Sodium"), (12, "Mg", "Magnesium"),
    (13, "Al", "Aluminum"), (14, "Si", "Silicon"), (15, "P", "Phosphorus"),
    (16, "S", "Sulfur"), (17, "Cl", "Chlorine"), (18, "Ar", "Argon"),
    (19, "K", "Potassium"), (20, "Ca", "Calcium"), (21, "Sc", "Scandium"),
    (22, "Ti", "Titanium"), (23, "V", "Vanadium"), (24, "Cr", "Chromium"),
    (25, "Mn", "Manganese"), (26, "Fe", "Iron"), (27, "Co", "Cobalt"),
    (28, "Ni", "Nickel"), (29, "Cu", "Copper"), (30, "Zn", "Zinc"),
    (31, "Ga", "Gallium"), (32, "Ge", "Germanium"), (33, "As", "Arsenic"),
    (34, "Se", "Selenium"), (35, "Br", "Bromine"), (36, "Kr", "Krypton"),
    (37, "Rb", "Rubidium"), (38, "Sr", "Strontium"), (39, "Y", "Yttrium"),
    (40, "Zr", "Zirconium"), (41, "Nb", "Niobium"), (42, "Mo", "Molybdenum"),
    (43, "Tc", "Technetium"), (44, "Ru", "Ruthenium"), (45, "Rh", "Rhodium"),
    (46, "Pd", "Palladium"), (47, "Ag", "Silver"), (48, "Cd", "Cadmium"),
    (49, "In", "Indium"), (50, "Sn", "Tin"), (51, "Sb", "Antimony"),
    (52, "Te", "Tellurium"), (53, "I", "Iodine"), (54, "Xe", "Xenon"),
    (55, "Cs", "Cesium"), (56, "Ba", "Barium"), (57, "La", "Lanthanum"),
    (58, "Ce", "Cerium"), (59, "Pr", "Praseodymium"), (60, "Nd", "Neodymium"),
    (61, "Pm", "Promethium"), (62, "Sm", "Samarium"), (63, "Eu", "Europium"),
    (64, "Gd", "Gadolinium"), (65, "Tb", "Terbium"), (66, "Dy", "Dysprosium"),
    (67, "Ho", "Holmium"), (68, "Er", "Erbium"), (69, "Tm", "Thulium"),
    (70, "Yb", "Ytterbium"), (71, "Lu", "Lutetium"), (72, "Hf", "Hafnium"),
    (73, "Ta", "Tantalum"), (74, "W", "Tungsten"), (75, "Re", "Rhenium"),
    (76, "Os", "Osmium"), (77, "Ir", "Iridium"), (78, "Pt", "Platinum"),
    (79, "Au", "Gold"), (80, "Hg", "Mercury"), (81, "Tl", "Thallium"),
    (82, "Pb", "Lead"), (83, "Bi", "Bismuth"), (84, "Po", "Polonium"),
    (85, "At", "Astatine"), (86, "Rn", "Radon"), (87, "Fr", "Francium"),
    (88, "Ra", "Radium"), (89, "Ac", "Actinium"), (90, "Th", "Thorium"),
    (91, "Pa", "Protactinium"), (92, "U", "Uranium"), (93, "Np", "Neptunium"),
    (94, "Pu", "Plutonium"), (95, "Am", "Americium"), (96, "Cm", "Curium"),
    (97, "Bk", "Berkelium"), (98, "Cf", "Californium"),
    (99, "Es", "Einsteinium"), (100, "Fm", "Fermium"),
    (101, "Md", "Mendelevium"), (102, "No", "Nobelium"),
    (103, "Lr", "Lawrencium"), (104, "Rf", "Rutherfordium"),
    (105, "Db", "Dubnium"), (106, "Sg", "Seaborgium"),
    (107, "Bh", "Bohrium"), (108, "Hs", "Hassium"),
    (109, "Mt", "Meitnerium"), (110, "Ds", "Darmstadtium"),
    (111, "Rg", "Roentgenium"), (112, "Cn", "Copernicium"),
    (113, "Nh", "Nihonium"), (114, "Fl", "Flerovium"),
    (115, "Mc", "Moscovium"), (116, "Lv", "Livermorium"),
    (117, "Ts", "Tennessine"), (118, "Og", "Oganesson"),
]


def _generate_science_pairs() -> List[Dict[str, str]]:
    """Three question styles per element: symbol, atomic number, name. ~354 pairs."""
    pairs: List[Dict[str, str]] = []
    for num, sym, name in ELEMENTS:
        pairs.append({
            "question": f"What is the chemical symbol for {name}?",
            "expected_answer_substring": sym,
            "category": "science",
        })
        pairs.append({
            "question": f"What is the atomic number of {name}?",
            "expected_answer_substring": str(num),
            "category": "science",
        })
        pairs.append({
            "question": f"Which element has atomic number {num}?",
            "expected_answer_substring": name,
            "category": "science",
        })
    return pairs


# ---------------------------------------------------------------------------
# 4. DATES — historical events (100+)
# ---------------------------------------------------------------------------

# (event_description, year_string)
HISTORICAL_EVENTS = [
    ("the fall of the Western Roman Empire", "476"),
    ("the signing of the Magna Carta", "1215"),
    ("the fall of Constantinople", "1453"),
    ("Columbus's first voyage to the Americas", "1492"),
    ("the start of the Protestant Reformation", "1517"),
    ("the defeat of the Spanish Armada", "1588"),
    ("the founding of Jamestown", "1607"),
    ("the Mayflower landing at Plymouth", "1620"),
    ("the English Civil War beginning", "1642"),
    ("the Great Fire of London", "1666"),
    ("the Glorious Revolution in England", "1688"),
    ("the start of the American Revolution", "1775"),
    ("the signing of the US Declaration of Independence", "1776"),
    ("the storming of the Bastille", "1789"),
    ("the start of the French Revolution", "1789"),
    ("the Battle of Trafalgar", "1805"),
    ("the Battle of Waterloo", "1815"),
    ("the Congress of Vienna", "1815"),
    ("Greek independence from the Ottoman Empire", "1829"),
    ("the abolition of slavery in the British Empire", "1833"),
    ("the start of the Mexican-American War", "1846"),
    ("the California Gold Rush beginning", "1848"),
    ("the publication of the Communist Manifesto", "1848"),
    ("the start of the Crimean War", "1853"),
    ("the start of the American Civil War", "1861"),
    ("the abolition of serfdom in Russia", "1861"),
    ("the Emancipation Proclamation", "1863"),
    ("the end of the American Civil War", "1865"),
    ("the assassination of Abraham Lincoln", "1865"),
    ("the completion of the Transcontinental Railroad", "1869"),
    ("the unification of Germany", "1871"),
    ("the unification of Italy", "1871"),
    ("the invention of the telephone by Alexander Graham Bell", "1876"),
    ("Thomas Edison's invention of the practical light bulb", "1879"),
    ("the Scramble for Africa beginning (Berlin Conference)", "1884"),
    ("the Spanish-American War", "1898"),
    ("the Boxer Rebellion in China", "1900"),
    ("the Wright Brothers' first powered flight", "1903"),
    ("the start of the Russo-Japanese War", "1904"),
    ("Einstein's publication of special relativity", "1905"),
    ("the sinking of the Titanic", "1912"),
    ("the start of World War I", "1914"),
    ("the assassination of Archduke Franz Ferdinand", "1914"),
    ("the Russian Revolution", "1917"),
    ("the end of World War I", "1918"),
    ("the signing of the Treaty of Versailles", "1919"),
    ("women's suffrage in the United States (19th Amendment)", "1920"),
    ("the founding of the Soviet Union", "1922"),
    ("the Wall Street Crash", "1929"),
    ("the start of the Great Depression", "1929"),
    ("Hitler becoming Chancellor of Germany", "1933"),
    ("the start of World War II", "1939"),
    ("the attack on Pearl Harbor", "1941"),
    ("D-Day (the Normandy landings)", "1944"),
    ("the dropping of the atomic bomb on Hiroshima", "1945"),
    ("the end of World War II", "1945"),
    ("the founding of the United Nations", "1945"),
    ("India's independence from Britain", "1947"),
    ("the founding of the State of Israel", "1948"),
    ("the start of the Korean War", "1950"),
    ("the discovery of DNA's double helix structure", "1953"),
    ("the launch of Sputnik", "1957"),
    ("the Cuban Revolution", "1959"),
    ("the construction of the Berlin Wall", "1961"),
    ("Yuri Gagarin's first human spaceflight", "1961"),
    ("the Cuban Missile Crisis", "1962"),
    ("the assassination of John F. Kennedy", "1963"),
    ("the March on Washington and MLK's 'I Have a Dream' speech", "1963"),
    ("the passage of the US Civil Rights Act", "1964"),
    ("the start of the Vietnam War (US involvement escalation)", "1965"),
    ("the Six-Day War", "1967"),
    ("the Apollo 11 Moon landing", "1969"),
    ("Woodstock music festival", "1969"),
    ("the Watergate scandal beginning", "1972"),
    ("the end of the Vietnam War", "1975"),
    ("the Iranian Revolution", "1979"),
    ("the Soviet invasion of Afghanistan", "1979"),
    ("the Chernobyl nuclear disaster", "1986"),
    ("the fall of the Berlin Wall", "1989"),
    ("the Tiananmen Square protests", "1989"),
    ("the dissolution of the Soviet Union", "1991"),
    ("the start of the Gulf War", "1991"),
    ("the end of apartheid in South Africa", "1994"),
    ("Nelson Mandela becoming President of South Africa", "1994"),
    ("the Rwandan genocide", "1994"),
    ("the handover of Hong Kong to China", "1997"),
    ("the September 11 attacks", "2001"),
    ("the start of the War in Afghanistan (US-led)", "2001"),
    ("the start of the Iraq War", "2003"),
    ("the Indian Ocean tsunami", "2004"),
    ("Hurricane Katrina", "2005"),
    ("the global financial crisis", "2008"),
    ("Barack Obama's inauguration as US President", "2009"),
    ("the Arab Spring beginning", "2010"),
    ("the Fukushima nuclear disaster", "2011"),
    ("the death of Osama bin Laden", "2011"),
    ("the Syrian Civil War beginning", "2011"),
    ("the annexation of Crimea by Russia", "2014"),
    ("the Brexit referendum", "2016"),
    ("the start of the COVID-19 pandemic", "2020"),
    ("the storming of the US Capitol", "2021"),
    ("Russia's invasion of Ukraine", "2022"),
    # Ancient events
    ("the construction of the Great Pyramid of Giza (approximate)", "2560 BC"),
    ("the founding of Rome (traditional date)", "753 BC"),
    ("the Battle of Marathon", "490 BC"),
    ("the death of Socrates", "399 BC"),
    ("Alexander the Great's conquest of Persia", "331 BC"),
    ("the assassination of Julius Caesar", "44 BC"),
    ("the eruption of Mount Vesuvius destroying Pompeii", "79"),
    ("the birth of Islam (Muhammad's first revelation)", "610"),
    ("the Battle of Hastings", "1066"),
    ("the start of the First Crusade", "1096"),
    ("the signing of the Treaty of Westphalia", "1648"),
]


def _generate_dates_pairs() -> List[Dict[str, str]]:
    """One question per historical event. ~110 pairs."""
    pairs: List[Dict[str, str]] = []
    for event, year in HISTORICAL_EVENTS:
        pairs.append({
            "question": f"In what year did {event} occur?",
            "expected_answer_substring": year,
            "category": "dates",
        })
    return pairs


# ---------------------------------------------------------------------------
# 5. GENERAL — miscellaneous factual trivia
# ---------------------------------------------------------------------------

GENERAL_FACTS = [
    ("How many continents are there?", "7"),
    ("How many oceans are there?", "5"),
    ("How many planets are in our solar system?", "8"),
    ("How many states are in the United States?", "50"),
    ("How many stripes are on the US flag?", "13"),
    ("How many stars are on the US flag?", "50"),
    ("How many teeth does an adult human have?", "32"),
    ("How many bones are in the adult human body?", "206"),
    ("How many chromosomes do humans have?", "46"),
    ("How many chambers does the human heart have?", "4"),
    ("How many days are in a leap year?", "366"),
    ("How many days are in a common year?", "365"),
    ("How many hours are in a day?", "24"),
    ("How many minutes are in an hour?", "60"),
    ("How many seconds are in a minute?", "60"),
    ("How many weeks are in a year?", "52"),
    ("How many months are in a year?", "12"),
    ("How many sides does a hexagon have?", "6"),
    ("How many sides does a pentagon have?", "5"),
    ("How many sides does an octagon have?", "8"),
    ("How many sides does a triangle have?", "3"),
    ("How many sides does a decagon have?", "10"),
    ("What is the speed of light in km/s (approximately)?", "300,000"),
    ("What is the boiling point of water in degrees Celsius?", "100"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
    ("How many legs does a spider have?", "8"),
    ("How many legs does an insect have?", "6"),
    ("How many wings does a bee have?", "4"),
    ("What is the largest organ in the human body?", "skin"),
    ("What is the smallest bone in the human body?", "stapes"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the smallest planet in our solar system?", "Mercury"),
    ("What is the closest planet to the Sun?", "Mercury"),
    ("What is the farthest planet from the Sun?", "Neptune"),
    ("What is the hottest planet in our solar system?", "Venus"),
    ("What is the largest moon of Saturn?", "Titan"),
    ("What is the largest moon of Jupiter?", "Ganymede"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("What is the chemical formula for carbon dioxide?", "CO2"),
    ("What is the chemical formula for glucose?", "C6H12O6"),
    ("What is the tallest mountain in the world?", "Everest"),
    ("What is the longest river in the world?", "Nile"),
    ("What is the largest desert in the world?", "Sahara"),
    ("What is the largest ocean on Earth?", "Pacific"),
    ("What is the deepest point in the ocean?", "Mariana"),
    ("What is the largest country by area?", "Russia"),
    ("What is the smallest country by area?", "Vatican"),
    ("What is the most populous country in the world?", "India"),
    ("What is the longest wall ever built?", "Great Wall"),
    ("How many keys does a standard piano have?", "88"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many strings does a standard violin have?", "4"),
    ("What is the hardest natural substance?", "diamond"),
    ("What is the most abundant element in the universe?", "hydrogen"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
    ("What percentage of Earth's surface is covered by water (approximately)?", "71"),
    ("How many colors are in a rainbow?", "7"),
    ("What is the speed of sound in air at sea level (approximately, in m/s)?", "343"),
    ("How many vertices does a cube have?", "8"),
    ("How many edges does a cube have?", "12"),
    ("How many faces does a cube have?", "6"),
    ("What is the value of pi to two decimal places?", "3.14"),
    ("What is the square root of 144?", "12"),
    ("What is the square root of 169?", "13"),
    ("What is the square root of 256?", "16"),
    ("How many cards are in a standard deck?", "52"),
    ("How many suits are in a standard deck of cards?", "4"),
    ("How many players are on a soccer team on the field?", "11"),
    ("How many players are on a basketball team on the court?", "5"),
    ("How many players are on a baseball team on the field?", "9"),
    ("How many holes are on a standard golf course?", "18"),
    ("What is the atomic number of gold?", "79"),
    ("What is the atomic number of iron?", "26"),
    ("What is the boiling point of water in Fahrenheit?", "212"),
    ("How many bits are in a byte?", "8"),
    ("How many bytes are in a kilobyte?", "1024"),
    ("What is the base of the binary number system?", "2"),
    ("What is the base of the hexadecimal number system?", "16"),
    ("What is the base of the decimal number system?", "10"),
    ("What is the base of the octal number system?", "8"),
]


def _generate_general_pairs() -> List[Dict[str, str]]:
    """Miscellaneous factual Q&A pairs. ~80 pairs."""
    return [
        {
            "question": q,
            "expected_answer_substring": a,
            "category": "general",
        }
        for q, a in GENERAL_FACTS
    ]


# ---------------------------------------------------------------------------
# MAIN — assemble and write the dataset
# ---------------------------------------------------------------------------

def generate_dataset() -> List[Dict[str, str]]:
    """Combine all categories into a single dataset, shuffle, and return."""
    all_pairs: List[Dict[str, str]] = []
    all_pairs.extend(_generate_math_pairs(400))
    all_pairs.extend(_generate_geography_pairs())
    all_pairs.extend(_generate_science_pairs())
    all_pairs.extend(_generate_dates_pairs())
    all_pairs.extend(_generate_general_pairs())

    # Shuffle for variety when iterating, but use a fixed seed for
    # reproducibility so the dataset is identical across runs.
    rng = random.Random(12345)
    rng.shuffle(all_pairs)
    return all_pairs


def main() -> None:
    dataset = generate_dataset()

    # Write to data/qa_dataset_1000.json
    out_path = pathlib.Path(__file__).resolve().parent.parent / "data" / "qa_dataset_1000.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Print summary statistics
    from collections import Counter
    cats = Counter(p["category"] for p in dataset)
    print(f"Generated {len(dataset)} QA pairs -> {out_path}")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Quick verification: load back and spot-check a few examples
    with open(out_path) as f:
        loaded = json.load(f)
    assert len(loaded) == len(dataset), "Round-trip length mismatch!"
    print(f"\nVerification: loaded {len(loaded)} pairs back from JSON.")

    # Show 5 random examples from different categories
    rng = random.Random(99)
    samples = rng.sample(loaded, min(5, len(loaded)))
    print("\nSample entries:")
    for s in samples:
        print(f"  [{s['category']}] Q: {s['question']}")
        print(f"           A: {s['expected_answer_substring']}")


if __name__ == "__main__":
    main()
