"""
Scrapy items for MIST training data collection.

Data structure matches WEB_SCRAPING_PROMPT.md specifications.
"""
import scrapy


class MistScrapedItem(scrapy.Item):
    """Scraped automotive diagnostic record for MIST training."""

    fault_codes = scrapy.Field()  # List[str] - P-codes, manufacturer codes
    obd_data = scrapy.Field()  # Dict[str, float] - OBD-II sensor readings
    vehicle_context = scrapy.Field()  # Dict - make, model, year, engine, mileage
    repair_summary = scrapy.Field()  # str - Concise summary of repair steps (50-200 words)
    outcome = scrapy.Field()  # str - success|failure|partial|unknown
    source_url = scrapy.Field()  # str - URL of source page
    source_type = scrapy.Field()  # str - forum|video|documentation|tsb
    record_type = scrapy.Field()  # str - fault_code | cause_to_solution

    # Optional fields (raw_text passed to LLM pipeline)
    raw_text = scrapy.Field()
    symptoms = scrapy.Field()
    diagnostic_steps = scrapy.Field()
    parts_used = scrapy.Field()
    cost = scrapy.Field()
    time_taken = scrapy.Field()
    follow_up = scrapy.Field()
    related_codes = scrapy.Field()
    confidence_score = scrapy.Field()
    llm_confidence = scrapy.Field()  # From LLM extraction pipeline
    heuristic_score = scrapy.Field()  # From validation pipeline
    timestamp = scrapy.Field()
