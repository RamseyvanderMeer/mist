"""
Forum platform configs: URLs, search patterns, pagination.

Used by forum_spider for multi-forum crawling and fault-code search.
"""

# Common fault codes to search (OBD-II P-codes + BMW-specific)
FAULT_CODES_TO_SEARCH = [
    # Misfire
    "P0300", "P0301", "P0302", "P0303", "P0304", "P0305", "P0306",
    # Fuel/air
    "P0171", "P0174", "P0420", "P0430", "P0442", "P0455", "P0507",
    # Cam/vanos
    "P0015", "P0016", "P0340", "P0341", "P0365", "P0366",
    # HPFP/injectors (BMW)
    "P0087", "P0088", "P0090", "P0091", "P0093",
    # BMW manufacturer codes
    "2A87", "2F2E", "29E0", "29E1", "2C27", "2C28", "2C29",
    "2A82", "2A98", "2DEC", "2DED", "3100",
]

# Forum configs: platform, base URL, search URL template, pagination
FORUM_CONFIGS = {
    "bimmerfest": {
        "platform": "xenforo",
        "base": "https://www.bimmerfest.com",
        "search_url": "https://www.bimmerfest.com/search/search?keywords={code}",
        "thread_pattern": "threads/",
        "forum_urls": [
            "https://www.bimmerfest.com/forums/engine-drivetrain.28/",
            "https://www.bimmerfest.com/forums/bmw-3-series-e90-e91-e92-e93.24/",
            "https://www.bimmerfest.com/forums/forced-induction.31/",
            "https://www.bimmerfest.com/forums/do-it-yourself-h-q.33/",
            "https://www.bimmerfest.com/forums/general-bmw-questions.29/",
        ],
        "pagination": "page-{n}",  # /forum.28/page-2
        "supports_search": True,
    },
    "e90post": {
        "platform": "vbulletin",
        "base": "https://www.e90post.com",
        "search_url": None,  # vBulletin search often returns form, not results
        "thread_pattern": "showthread",
        "forum_urls": [
            "https://www.e90post.com/forums/forumdisplay.php?f=58",
            "https://www.e90post.com/forums/forumdisplay.php?f=16",
            "https://www.e90post.com/forums/forumdisplay.php?f=17",
            "https://www.e90post.com/forums/forumdisplay.php?f=95",
            "https://www.e90post.com/forums/forumdisplay.php?f=44",
        ],
        "pagination": "page={n}",  # &page=2
        "supports_search": False,
    },
}
