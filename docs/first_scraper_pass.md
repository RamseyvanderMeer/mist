 'timestamp': '2026-02-18T12:04:13.435034Z',
 'vehicle_context': {'make': 'Bmw', 'year': 2009}}
2026-02-18 04:04:15 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/fs-2003-bmw-325xi-touring-sport-5-speed-manual-3100.1286528/> (referer: https://www.bimmerfest.com/search/3074714/?page=2&q=3100&c%5BsearchProfileName%5D=control&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:16 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:16 [scrapy.core.scraper] WARNING: Dropped: Cause-to-solution requires outcome or symptoms
{'confidence_score': 0.5,
 'fault_codes': [],
 'llm_confidence': None,
 'obd_data': {'coolant_temp': 2018.0,
              'fuel_trim_bank1': 50.0,
              'fuel_trim_bank2': 50.0,
              'intake_air_temp': 2018.0,
              'throttle_position': 50.0},
 'outcome': 'unknown',
 'repair_summary': 'Replaced headers & catalytic converters - Nov 2017 The new '
                   'headers/cats are 50-state legal and have a 5 year warranty '
                   'Both front airbag recalls are done Michelin Pilot Sport '
                   'A/S 3 Tires - March 2015 @ 183K Preventative total cooling '
                   'system replacement - Dec 2013 View the full maintenance '
                   'log. Receipts available upon request.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/fs-2003-bmw-325xi-touring-sport-5-speed-manual-3100.1286528/',
 'timestamp': '2026-02-18T12:04:16.008029Z',
 'vehicle_context': {'make': 'Bmw', 'mileage': 200250, 'year': 2003}}
2026-02-18 04:04:18 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/taz-roatrip-version-2-0.1456005/> (referer: https://www.bimmerfest.com/search/3074714/?page=2&q=3100&c%5BsearchProfileName%5D=control&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:20 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:21 [scrapers.pipelines.postgres] DEBUG: Wrote item to Postgres: https://www.bimmerfest.com/threads/taz-roatrip-version-2-0.1
2026-02-18 04:04:21 [scrapy.core.scraper] DEBUG: Scraped from <200 https://www.bimmerfest.com/threads/taz-roatrip-version-2-0.1456005/>
{'confidence_score': 0.55,
 'fault_codes': [],
 'heuristic_score': 0.55,
 'llm_confidence': None,
 'obd_data': {'fuel_trim_bank1': 99.0,
              'fuel_trim_bank2': 99.0,
              'throttle_position': 99.0},
 'outcome': 'success',
 'record_type': 'cause_to_solution',
 'repair_summary': 'As every summer for the last 4 years, the ending of July '
                   'or the beginning of August, is my vacation period and road '
                   'trip time for TAZ. Unfortunately, last year, we renovated '
                   'our summer residence, and that killed all of our time, and '
                   'I mean ALL of it( we finished 2 days before leaving for '
                   'Denmark). Therefore, no post was created, but TAZ did the '
                   'usual 5000km( 3100 miles) with a small hiccup. The '
                   'alternator( original) died 30 km after departure.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/taz-roatrip-version-2-0.1456005/',
 'timestamp': '2026-02-18T12:04:18.814069Z',
 'vehicle_context': {'engine': 'LS3',
                     'make': 'Bmw',
                     'mileage': 5000,
                     'year': 2009}}
2026-02-18 04:04:21 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/putting-out-feelers-for-my-2019-bmw-m240i.1459003/> (referer: https://www.bimmerfest.com/search/3074714/?page=2&q=3100&c%5BsearchProfileName%5D=control&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:22 [scrapers.pipelines.langgraph_pipeline] DEBUG: LLM extracted: Replaced rear tail light assembly under warranty. (confidence 1.00)
2026-02-18 04:04:22 [scrapy.core.scraper] WARNING: Dropped: No valid fault codes and no substantial cause-to-solution
{'confidence_score': 1.0,
 'fault_codes': [],
 'llm_confidence': 1.0,
 'obd_data': {'coolant_temp': 0.0, 'intake_air_temp': 0.0},
 'outcome': 'success',
 'repair_summary': 'Replaced rear tail light assembly under warranty.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/putting-out-feelers-for-my-2019-bmw-m240i.1459003/',
 'symptoms': 'faulty rear tail light assembly',
 'timestamp': '2026-02-18T12:04:21.618573Z',
 'vehicle_context': {'make': 'BMW', 'model': 'M240i', 'year': '2019'}}
2026-02-18 04:04:23 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/need-help-figuring-something-out-n54.1456756/> (referer: https://www.bimmerfest.com/search/3074714/?page=2&q=3100&c%5BsearchProfileName%5D=control&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:24 [scrapers.pipelines.langgraph_pipeline] DEBUG: Skipping LLM (no fault codes or repair-like text in content)
2026-02-18 04:04:24 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:24 [scrapy.core.scraper] WARNING: Dropped: Cause-to-solution requires outcome or symptoms
{'confidence_score': 0.5,
 'fault_codes': [],
 'llm_confidence': None,
 'obd_data': {'coolant_temp': 54.0,
              'engine_rpm': 4500.0,
              'intake_air_temp': 54.0},
 'outcome': 'unknown',
 'repair_summary': 'I have a 2007 335i sedan mt, fbo, pure 600 turbos, jb4, im '
                   'having problems with when i hit the gas it fees pretty '
                   'slow until about 4500 rpms, then it almost feels like i '
                   'instantly hit a 250 shot of nitrous for a second, then it '
                   'goes into low boost mode, with error code 30fe boost over '
                   'target and 3100 low boost mode engaged- cel displayed. i '
                   'have no clue whats wrong with it, any advice, just put in '
                   'new coils and plugs, upgraded dme to the msd 81.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/need-help-figuring-something-out-n54.1456756/',
 'timestamp': '2026-02-18T12:04:24.044554Z',
 'vehicle_context': {'year': 2007}}
2026-02-18 04:04:24 [scrapy.extensions.logstats] INFO: Crawled 4712 pages (at 24 pages/min), scraped 3317 items (at 5 items/min)
2026-02-18 04:04:24 [scrapy.extensions.logstats] INFO: Crawled 4712 pages (at 0 pages/min), scraped 3317 items (at 0 items/min)
2026-02-18 04:04:26 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/cam-sensor.1454232/> (referer: https://www.bimmerfest.com/search/3074714/?page=2&q=3100&c%5BsearchProfileName%5D=control&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:27 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:27 [scrapy.core.scraper] WARNING: Dropped: Quality score 0.35 below threshold 0.5
{'confidence_score': 0.35,
 'fault_codes': [],
 'heuristic_score': 0.35,
 'llm_confidence': None,
 'obd_data': {},
 'outcome': 'success',
 'record_type': 'cause_to_solution',
 'repair_summary': 'My car threw a cam sensor code yesterday along with 3100 '
                   'boost reduction. I cleared the code everything is back to '
                   'normal. My question is are the Intake and exhaust Cam '
                   'sensor the same and interchangeable. My car threw a cam '
                   'sensor code yesterday along with 3100 boost reduction. I '
                   'cleared the code everything is back to normal. My question '
                   'is are the Intake and exhaust Cam sensor the same and '
                   'interchangeable. My car threw a cam sensor code yesterday '
                   'along with 3100 boost reduction.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/cam-sensor.1454232/',
 'timestamp': '2026-02-18T12:04:26.913181Z',
 'vehicle_context': {'year': 2004}}
2026-02-18 04:04:29 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/please-help-650i-bmw.1479392/> (referer: https://www.bimmerfest.com/search/3074714/?q=3100&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:29 [scrapers.pipelines.langgraph_pipeline] DEBUG: Skipping LLM (no fault codes or repair-like text in content)
2026-02-18 04:04:29 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:29 [scrapy.core.scraper] WARNING: Dropped: Cause-to-solution requires outcome or symptoms
{'confidence_score': 0.5,
 'fault_codes': [],
 'llm_confidence': None,
 'obd_data': {'coolant_temp': 300.0, 'intake_air_temp': 300.0},
 'outcome': 'unknown',
 'repair_summary': 'Can anyone help me out. I have a EWS code that comes up to '
                   'F44 says manipulation car cranks but won’t start all I '
                   'have is O2 scanner end of a 3100 and the ANCELBB 300 can '
                   'anyone help me figure out what to do Please and thank you\n'
                   'Can anyone help me out. I have a EWS code that comes up to '
                   'F44 says manipulation car cranks but won’t start all I '
                   'have is O2 scanner end of a 3100 and the ANCELBB 300 can '
                   'anyone help me figure out what to do Please and thank you\n'
                   'Can anyone help me out.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/please-help-650i-bmw.1479392/',
 'timestamp': '2026-02-18T12:04:29.311703Z',
 'vehicle_context': {'year': 2025}}
2026-02-18 04:04:32 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/2e7c-3100-after-new-alt-and-cps.1391639/> (referer: https://www.bimmerfest.com/search/3074714/?q=3100&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:32 [scrapers.pipelines.langgraph_pipeline] DEBUG: Skipping LLM (no fault codes or repair-like text in content)
2026-02-18 04:04:32 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:32 [scrapy.core.scraper] WARNING: Dropped: Cause-to-solution requires outcome or symptoms
{'confidence_score': 0.5,
 'fault_codes': [],
 'llm_confidence': None,
 'obd_data': {'coolant_temp': 7.0,
              'fuel_trim_bank1': 80.0,
              'fuel_trim_bank2': 80.0,
              'intake_air_temp': 7.0,
              'throttle_position': 80.0},
 'outcome': 'unknown',
 'repair_summary': 'I have an 08 335i N54 that has now been through a new '
                   'alternator, new cps, new battery, and is now throwing '
                   "2E7C, 3100. I'm hoping you guys can help me out, let me "
                   'know what info you need. Car has Eldors and NGK 1 Step '
                   'Colders. I***8217;m tuned MHD STAGE 2+\n'
                   'I have an 08 335i N54 that has now been through a new '
                   'alternator, new cps, new battery, and is now throwing '
                   "2E7C, 3100. I'm hoping you guys can help me out, let me "
                   'know what info you need. Car has Eldors and NGK 1 Step '
                   'Colders.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/2e7c-3100-after-new-alt-and-cps.1391639/',
 'timestamp': '2026-02-18T12:04:32.325418Z',
 'vehicle_context': {'engine': 'N54', 'make': 'Bmw', 'year': 2015}}
2026-02-18 04:04:34 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/2a7a-3100-limp-mode.1419338/> (referer: https://www.bimmerfest.com/search/3074714/?q=3100&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:36 [scrapers.pipelines.langgraph_pipeline] DEBUG: LLM extracted: Reversed VANOS plugs. (confidence 1.00)
2026-02-18 04:04:36 [scrapy.core.scraper] WARNING: Dropped: No valid fault codes and no substantial cause-to-solution
{'confidence_score': 1.0,
 'fault_codes': ['2A7A', '3100'],
 'llm_confidence': 1.0,
 'obd_data': {},
 'outcome': 'success',
 'repair_summary': 'Reversed VANOS plugs.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/2a7a-3100-limp-mode.1419338/',
 'symptoms': 'Idles great for 20 seconds before half engine light shows',
 'timestamp': '2026-02-18T12:04:34.985828Z',
 'vehicle_context': {}}
2026-02-18 04:04:36 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/cant-clear-error-3100.1419782/> (referer: https://www.bimmerfest.com/search/3074714/?q=3100&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:37 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=[]
2026-02-18 04:04:37 [scrapy.core.scraper] WARNING: Dropped: Cause-to-solution requires outcome or symptoms
{'confidence_score': 0.5,
 'fault_codes': [],
 'llm_confidence': None,
 'obd_data': {'coolant_temp': 2012.0, 'intake_air_temp': 2012.0},
 'outcome': 'unknown',
 'repair_summary': "reset in both mhd and protools doesn't clear 3100. What's "
                   'the next step in diagnostic? I understand it might '
                   'possibly be turbo solenoids or vanos solenoids how do I '
                   'check? Thx\n'
                   'Hi - picked up limp mode 3100 yesterday  (shows CD29 in '
                   'dash) .',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/cant-clear-error-3100.1419782/',
 'timestamp': '2026-02-18T12:04:36.978222Z',
 'vehicle_context': {'engine': 'N54', 'make': 'Bmw', 'year': 2014}}
2026-02-18 04:04:39 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/2abc-and-3100-codes-always-showing.1404633/> (referer: https://www.bimmerfest.com/search/3074714/?q=3100&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:40 [scrapers.pipelines.langgraph_pipeline] DEBUG: LLM extracted: Changed the charge pressure sensor. (confidence 0.90)
2026-02-18 04:04:40 [scrapy.core.scraper] WARNING: Dropped: No valid fault codes and no substantial cause-to-solution
{'confidence_score': 0.9,
 'fault_codes': ['2ABC', '3100'],
 'llm_confidence': 0.9,
 'obd_data': {'coolant_temp': 3100.0, 'intake_air_temp': 3100.0},
 'outcome': 'failure',
 'repair_summary': 'Changed the charge pressure sensor.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/2abc-and-3100-codes-always-showing.1404633/',
 'symptoms': 'Limp mode, misfiring in multiple cylinders',
 'timestamp': '2026-02-18T12:04:39.205894Z',
 'vehicle_context': {'make': 'BMW', 'model': '335i'}}
2026-02-18 04:04:41 [scrapy.core.engine] DEBUG: Crawled (200) <GET https://www.bimmerfest.com/threads/2009-535i-crank-no-start.1477282/> (referer: https://www.bimmerfest.com/search/3074714/?q=3100&c%5BshowFilter%5D=visibleOnly&o=relevance)
2026-02-18 04:04:43 [scrapers.pipelines.langgraph_pipeline] DEBUG: Regex fallback: fault_codes=['K6300', 'X8680']
2026-02-18 04:04:47 [scrapers.pipelines.postgres] DEBUG: Wrote item to Postgres: https://www.bimmerfest.com/threads/2009-535i-crank-no-start.
2026-02-18 04:04:47 [scrapy.core.scraper] DEBUG: Scraped from <200 https://www.bimmerfest.com/threads/2009-535i-crank-no-start.1477282/>
{'confidence_score': 0.8,
 'fault_codes': ['K6300', 'X8680'],
 'heuristic_score': 0.8,
 'llm_confidence': None,
 'obd_data': {'engine_rpm': 100.0},
 'outcome': 'success',
 'record_type': 'fault_code',
 'repair_summary': 'replaced crank and camshaft sensors, cleaned vanos '
                   'solenoids, replaced the fuel pump. He told me intake '
                   "valves looked clean from back side (they weren't bad) and "
                   'appears he or someone used RTV as thee valve cover gasket.',
 'source_type': 'forum',
 'source_url': 'https://www.bimmerfest.com/threads/2009-535i-crank-no-start.1477282/',
 'timestamp': '2026-02-18T12:04:42.225758Z',
 'vehicle_context': {'make': 'Bmw', 'year': 2009}}
2026-02-18 04:04:47 [scrapy.core.engine] INFO: Closing spider (finished)
2026-02-18 04:04:47 [scrapers.pipelines.langgraph_pipeline] INFO: LLM token usage: 3924 calls, 26304930 input + 384662 output = 26689592 total tokens, est. cost $2.784358
2026-02-18 04:04:47 [scrapy.statscollectors] INFO: Dumping Scrapy stats:
{'downloader/request_bytes': 4594114,
 'downloader/request_count': 4755,
 'downloader/request_method_count/GET': 4755,
 'downloader/response_bytes': 295443165,
 'downloader/response_count': 4755,
 'downloader/response_status_count/200': 4719,
 'downloader/response_status_count/303': 36,
 'elapsed_time_seconds': 13103.168422,
 'finish_reason': 'finished',
 'finish_time': datetime.datetime(2026, 2, 18, 12, 4, 47, 421786, tzinfo=datetime.timezone.utc),
 'httpcompression/response_bytes': 1835373351,
 'httpcompression/response_count': 4719,
 'item_dropped_count': 953,
 'item_dropped_reasons_count/DropItem': 953,
 'item_scraped_count': 3318,
 'items_per_minute': 15.193467144928643,
 'log_count/DEBUG': 15592,
 'log_count/INFO': 595,
 'log_count/WARNING': 1035,
 'memusage/max': 259325952,
 'memusage/startup': 128172032,
 'request_depth_max': 25,
 'response_received_count': 4719,
 'responses_per_minute': 21.6087918797222,
 'scheduler/dequeued': 4755,
 'scheduler/dequeued/memory': 4755,
 'scheduler/enqueued': 4755,
 'scheduler/enqueued/memory': 4755,
 'start_time': datetime.datetime(2026, 2, 18, 8, 26, 24, 253364, tzinfo=datetime.timezone.utc)}
2026-02-18 04:04:47 [scrapy.core.engine] INFO: Spider closed (finished)