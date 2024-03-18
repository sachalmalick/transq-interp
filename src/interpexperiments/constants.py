ATTENDERS_DIR = "output/attenders"

PROMPT_1 = "if all men are humans and all humans suck it can be inferred that all men "
PROMPT_2 = "if all men are humans and all humans eat it can be inferred that all men "
PROMPT_3 = "if all men are mammals and all mammals eat it can be inferred that all men "
PROMPT_4 = "if all dogs are mammals and all mammals eat it can be inferred that all dogs "
PROMPT_5 = "if all dogs are mammals and all mammals eat it can be deduced that all dogs "
PROMPT_6 = "if all dogs are mammals and all mammals eat it can be deduced that dogs do not "
PROMPT_7 = "if all dogs are friendly and friendly is a type of kind then it can be inferred that all dogs are a type of "
PROMPT_8 = "dog implies friendly and if friendly then kind therefore by the transitive property dog also implies "
PROMPT_9 = "yoda implies ninja and if ninja then dangerous therefore by the transitive property ninja also implies "
PROMPT_10 = "yoda implies ninja and if ninja then dangerous therefore by the transitive property coco also implies "

PROMPTS = {"prompt_1" : PROMPT_1, "prompt_2" : PROMPT_2 , "prompt_3" : PROMPT_3, "prompt_4" : PROMPT_4, "prompt_5" : PROMPT_5, "prompt_6" : PROMPT_6, "prompt_7" : PROMPT_7, "prompt_8" : PROMPT_8, "prompt_9" : PROMPT_9, "prompt_10" : PROMPT_10}

WEIGHT_DIFFS_DIR = "output/weightdiffs"

FIRST = [
    "cat", "dog", "house", "car", "tree", "bird", "table", "chair", "book", "computer",
    "phone", "flower", "pen", "pencil", "paper", "river", "mountain", "ocean", "lake", "sun",
    "moon", "star", "planet", "city", "country", "bicycle", "boat", "ship", "airplane", "train",
    "bus", "elephant", "lion", "tiger", "bear", "wolf", "fox", "rabbit", "horse", "snake", "fish",
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "melon", "pineapple", "pear",
    "peach", "plum", "kiwi", "mango", "cherry", "lemon", "lime", "coconut", "avocado", "potato",
    "tomato", "onion", "garlic", "lettuce", "cucumber", "carrot", "broccoli", "pepper", "mushroom",
    "eggplant", "corn", "spinach", "asparagus", "celery", "peas", "beans", "rice", "pasta", "bread",
    "cheese", "milk", "yogurt", "butter", "egg", "chicken", "beef", "pork", "fish", "shrimp", "lobster",
    "crab", "salad", "soup", "sandwich", "pizza", "burger", "fries", "cake", "cookie", "ice cream", "chocolate"
]

SECOND = [
    "desk", "lamp", "mirror", "bed", "window", "door", "key", "lock", "wallet", "bag",
    "umbrella", "jacket", "shirt", "pants", "socks", "shoes", "hat", "glasses", "watch", "bracelet",
    "ring", "necklace", "earrings", "television", "remote", "speaker", "microwave", "oven", "refrigerator",
    "sink", "toilet", "mirror", "brush", "comb", "shampoo", "soap", "towel", "laptop", "tablet", "keyboard",
    "mouse", "headphones", "charger", "battery", "clock", "calendar", "wallet", "coin", "bank", "wallet",
    "coin", "bank", "credit card", "debit card", "cash", "receipt", "ticket", "map", "compass", "camera",
    "binoculars", "telescope", "microscope", "guitar", "piano", "violin", "drums", "trumpet", "flute",
    "saxophone", "harmonica", "accordion", "banjo", "ukulele", "harp", "painting", "sculpture", "drawing",
    "photograph", "pottery", "vase", "statue", "figurine", "ornament", "artifact", "tapestry", "rug",
    "curtain", "pillow", "blanket", "quilt", "mattress", "candle", "incense", "perfume", "fragrance", "lotion"
]

THIRD = [
    "telephone", "mailbox", "backpack", "sunglasses", "umbrella", "raincoat", "wallet", "briefcase", "suitcase", "watch",
    "earbuds", "headphones", "headset", "thermometer", "thermostat", "umbrella", "canteen", "thermos", "snorkel", "swimsuit",
    "flippers", "raft", "lifejacket", "tent", "sleeping bag", "campfire", "binoculars", "backpack", "flashlight", "compass",
    "map", "treasure", "scroll", "treasure chest", "dagger", "sword", "shield", "armor", "crown", "throne",
    "wand", "staff", "potion", "scroll", "book", "spellbook", "amulet", "ring", "gemstone", "crystal",
    "bracelet", "necklace", "earrings", "candlestick", "vase", "statue", "painting", "portrait", "talisman", "scarab",
    "coin", "medallion", "artifact", "fountain", "gargoyle", "gate", "statue", "arch", "castle", "moat",
    "drawbridge", "dungeon", "tower", "fortress", "barricade", "rampart", "siege", "catapult", "ballista", "trebuchet",
    "battering ram", "ladder", "rope", "torch", "brazier", "cauldron", "grimoire", "alchemy", "potion", "elixir"
]

