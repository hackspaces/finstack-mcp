"""
FinStack Thematic Baskets

Curated thematic/micro-sector baskets of NSE tickers. Official Nifty sectoral
indices are only ~15; these add the NICHE buckets (pipes, HVAC, refrigerants,
specialty chemicals, defense, railways, EV, CDMO, ...) that no free index
provides. A basket "index" is computed on the fly (equal-weight) from the
constituents' yfinance prices.

Tickers are bare NSE symbols (".NS" appended at fetch time). The set is
auto-validated against yfinance (see scripts) and degrades gracefully — a
basket reports which constituents resolved. The `sector` tool also accepts an
arbitrary symbol list, so Claude can compose any ad-hoc/combinatorial bucket.
"""

BASKETS: dict[str, dict] = {
    # ── Pipes / plumbing / building materials ──
    "pipes_plumbing": {"category": "building_materials", "symbols": ["SUPREMEIND", "ASTRAL", "FINPIPE", "PRINCEPIPE", "APOLLOPIPE", "JINDALPOLY"]},
    "tiles_ceramics": {"category": "building_materials", "symbols": ["KAJARIACER", "SOMANYCERA", "CERA", "ORIENTBELL", "HSIL"]},
    "sanitaryware_bath": {"category": "building_materials", "symbols": ["CERA", "HSIL", "SOMANYCERA"]},
    "plywood_laminates": {"category": "building_materials", "symbols": ["CENTURYPLY", "GREENPLY", "GREENLAM", "RUSHIL"]},
    "paints": {"category": "building_materials", "symbols": ["ASIANPAINT", "BERGEPAINT", "KANSAINER", "AKZOINDIA", "INDIGOPNTS"]},
    "adhesives": {"category": "building_materials", "symbols": ["PIDILITIND", "ASTRAL"]},
    "cement": {"category": "materials", "symbols": ["ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALBHARAT", "JKCEMENT", "RAMCOCEM", "JKLAKSHMI", "HEIDELBERG", "STARCEMENT"]},

    # ── HVAC / cooling / consumer durables ──
    "hvac_cooling": {"category": "consumer_durables", "symbols": ["VOLTAS", "BLUESTARCO", "SYMPHONY", "AMBER"]},
    "consumer_durables": {"category": "consumer_durables", "symbols": ["HAVELLS", "VOLTAS", "WHIRLPOOL", "CROMPTON", "VGUARD", "BAJAJELEC", "ORIENTELEC", "TTKPRESTIG", "BLUESTARCO", "SYMPHONY"]},
    "electronics_mfg_emS": {"category": "electronics", "symbols": ["DIXON", "AMBER", "KAYNES", "SYRMA", "PGEL", "CYIENTDLM"]},
    "cables_wires": {"category": "electricals", "symbols": ["POLYCAB", "KEI", "FINCABLES", "RRKABEL", "HAVELLS"]},
    "transformers_power_equip": {"category": "electricals", "symbols": ["TRANSWORLD", "VOLTAMP", "TARIL", "GVT&D", "HBLENGINE"]},

    # ── Chemicals (sub-themes) ──
    "specialty_chemicals": {"category": "chemicals", "symbols": ["SRF", "PIIND", "AARTIIND", "VINATIORGA", "NAVINFLUOR", "ATUL", "DEEPAKNTR", "CLEAN", "FINEORG", "GALAXYSURF"]},
    "fluorochemicals_refrigerants": {"category": "chemicals", "symbols": ["NAVINFLUOR", "SRF", "GFLLIMITED", "FLUOROCHEM"]},
    "agrochemicals": {"category": "chemicals", "symbols": ["UPL", "PIIND", "BAYERCROP", "RALLIS", "SUMICHEM", "DHANUKA", "INSECTICID", "BASF"]},
    "fertilizers": {"category": "chemicals", "symbols": ["COROMANDEL", "CHAMBLFERT", "GNFC", "GSFC", "DEEPAKFERT", "RCF", "NFL", "FACT", "PARADEEP"]},
    "commodity_chemicals": {"category": "chemicals", "symbols": ["TATACHEM", "GNFC", "GHCL", "DCW", "NOCIL", "ALKYLAMINE", "BALAMINES"]},
    "paper": {"category": "materials", "symbols": ["JKPAPER", "WSTCSTPAPR", "TNPL", "SESHAPAPER", "ANDHRAPAP"]},

    # ── Sugar / agri / food ──
    "sugar": {"category": "agri", "symbols": ["BALRAMCHIN", "TRIVENI", "DALMIASUG", "BAJAJHIND", "DHAMPURSUG", "EIDPARRY", "RENUKA", "AVADHSUGAR"]},
    "edible_oil_fmcg_agri": {"category": "agri", "symbols": ["MARICO", "AWL", "PATANJALI", "GODREJAGRO"]},
    "aquaculture_seafood": {"category": "agri", "symbols": ["AVANTIFEED", "APEX", "WATERBASE", "COASTCORP"]},
    "tea_coffee_plantation": {"category": "agri", "symbols": ["TATACONSUM", "CCL", "MCLEODRUSS", "JAYSREETEA"]},

    # ── Auto & ancillaries ──
    "auto_oems": {"category": "auto", "symbols": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY"]},
    "auto_ancillaries": {"category": "auto", "symbols": ["BOSCHLTD", "MOTHERSON", "BHARATFORG", "BALKRISIND", "MRF", "APOLLOTYRE", "SONACOMS", "ENDURANCE", "EXIDEIND", "UNOMINDA"]},
    "tyres": {"category": "auto", "symbols": ["MRF", "APOLLOTYRE", "BALKRISIND", "CEATLTD", "JKTYRE", "TVSSRICHAK"]},
    "ev_value_chain": {"category": "auto", "symbols": ["TATAMOTORS", "M&M", "TVSMOTOR", "OLAELEC", "SONACOMS", "EXIDEIND", "ARE&M", "HBLENGINE"]},
    "batteries": {"category": "auto", "symbols": ["EXIDEIND", "ARE&M", "HBLENGINE"]},
    "bearings": {"category": "industrials", "symbols": ["SKFINDIA", "SCHAEFFLER", "TIMKEN", "NRBBEARING"]},

    # ── Capital goods / industrials / defense / rail ──
    "capital_goods": {"category": "industrials", "symbols": ["LT", "SIEMENS", "ABB", "BHEL", "CGPOWER", "THERMAX", "BEL", "HAL", "KEC", "KPIL"]},
    "defense": {"category": "defense", "symbols": ["HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "DATAPATTNS", "PARAS", "ZENTEC", "MTARTECH", "ASTRAMICRO"]},
    "railways": {"category": "railways", "symbols": ["IRCTC", "IRFC", "RVNL", "IRCON", "RAILTEL", "TITAGARH", "JWL", "TEXRAIL", "RITES", "CONCOR"]},
    "shipbuilding": {"category": "defense", "symbols": ["MAZDOCK", "COCHINSHIP", "GRSE"]},
    "industrial_automation": {"category": "industrials", "symbols": ["SIEMENS", "ABB", "HONAUT"]},

    # ── Logistics / ports / shipping ──
    "logistics": {"category": "logistics", "symbols": ["CONCOR", "BLUEDART", "TCIEXP", "DELHIVERY", "MAHLOG", "VRLLOG", "GATI", "ALLCARGO", "TCI"]},
    "ports": {"category": "logistics", "symbols": ["ADANIPORTS", "JSWINFRA", "GPPL"]},
    "shipping": {"category": "logistics", "symbols": ["GESHIP", "SCI", "SEAMECLTD"]},

    # ── Pharma / healthcare sub-themes ──
    "pharma_large": {"category": "pharma", "symbols": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM", "ZYDUSLIFE", "MANKIND"]},
    "cdmo_cram": {"category": "pharma", "symbols": ["DIVISLAB", "LAURUSLABS", "SYNGENE", "GLAND", "NEULANDLAB", "SUVENPHAR", "COHANCE"]},
    "api_bulk_drugs": {"category": "pharma", "symbols": ["LAURUSLABS", "GRANULES", "AARTIDRUGS", "SOLARA", "NEULANDLAB", "DIVISLAB"]},
    "hospitals": {"category": "healthcare", "symbols": ["APOLLOHOSP", "MAXHEALTH", "FORTIS", "NH", "MEDANTA", "RAINBOW", "KIMS", "ASTERDM", "JLHL"]},
    "diagnostics": {"category": "healthcare", "symbols": ["LALPATHLAB", "METROPOLIS", "THYROCARE", "VIJAYA", "KRSNAA"]},

    # ── Financials sub-themes ──
    "private_banks": {"category": "financials", "symbols": ["HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK"]},
    "psu_banks": {"category": "financials", "symbols": ["SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "MAHABANK", "CENTRALBK"]},
    "small_finance_banks": {"category": "financials", "symbols": ["AUBANK", "EQUITASBNK", "UJJIVANSFB", "SURYODAY", "ESAFSFB"]},
    "nbfc": {"category": "financials", "symbols": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "SHRIRAMFIN", "M&MFIN", "LICHSGFIN", "PEL", "SUNDARMFIN", "POONAWALLA"]},
    "housing_finance": {"category": "financials", "symbols": ["LICHSGFIN", "PNBHOUSING", "HOMEFIRST", "AAVAS", "APTUS", "CANFINHOME"]},
    "microfinance": {"category": "financials", "symbols": ["CREDITACC", "FUSION", "SPANDANA", "UJJIVANSFB"]},
    "insurance": {"category": "financials", "symbols": ["SBILIFE", "HDFCLIFE", "ICICIPRULI", "ICICIGI", "LICI", "STARHEALTH", "MFSL", "NIACL", "GICRE"]},
    "amc_wealth": {"category": "financials", "symbols": ["HDFCAMC", "NAM-INDIA", "UTIAMC", "ABSLAMC", "360ONE", "ANANDRATHI", "PRUDENT"]},
    "capital_market_infra": {"category": "financials", "symbols": ["BSE", "MCX", "CDSL", "CAMS", "KFINTECH", "ANGELONE", "IEX", "MOTILALOFS", "NUVAMA"]},
    "fintech_newage": {"category": "financials", "symbols": ["PAYTM", "POLICYBZR", "ANGELONE", "BSE", "CDSL"]},

    # ── Consumer / discretionary ──
    "fmcg": {"category": "consumer", "symbols": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "EMAMILTD"]},
    "qsr_food": {"category": "consumer", "symbols": ["JUBLFOOD", "DEVYANI", "SAPPHIRE", "WESTLIFE", "RBA"]},
    "jewellery": {"category": "consumer", "symbols": ["TITAN", "KALYANKJIL", "SENCO", "PCJEWELLER", "THANGAMAYL"]},
    "footwear": {"category": "consumer", "symbols": ["BATAINDIA", "RELAXO", "METROBRAND", "CAMPUS"]},
    "retail": {"category": "consumer", "symbols": ["DMART", "TRENT", "ABFRL", "VMART", "SHOPERSTOP", "MANYAVAR"]},
    "hotels_tourism": {"category": "consumer", "symbols": ["INDHOTEL", "EIHOTEL", "CHALET", "LEMONTREE", "MHRIL", "ITCHOTELS"]},
    "alcohol_beverages": {"category": "consumer", "symbols": ["UNITDSPR", "RADICO", "UBL", "GLOBUSSPR"]},
    "textiles": {"category": "consumer", "symbols": ["PAGEIND", "KPRMILL", "TRIDENT", "WELSPUNLIV", "VARDHACRLC", "GOKEX", "ARVIND"]},

    # ── Energy / utilities / materials ──
    "oil_gas": {"category": "energy", "symbols": ["RELIANCE", "ONGC", "IOC", "BPCL", "HINDPETRO", "GAIL", "OIL", "PETRONET"]},
    "city_gas_distribution": {"category": "energy", "symbols": ["IGL", "MGL", "GUJGASLTD", "ATGL", "GSPL"]},
    "power_utilities": {"category": "utilities", "symbols": ["NTPC", "POWERGRID", "TATAPOWER", "JSWENERGY", "NHPC", "SJVN", "TORNTPOWER", "CESC", "ADANIPOWER", "ADANIENSOL"]},
    "renewables_solar": {"category": "utilities", "symbols": ["ADANIGREEN", "SUZLON", "INOXWIND", "WAAREEENER", "PREMIERENE", "BORORENEW"]},
    "power_financiers": {"category": "financials", "symbols": ["PFC", "RECLTD", "IREDA", "PTC"]},
    "metals_mining": {"category": "metals", "symbols": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL", "SAIL", "NMDC", "NATIONALUM", "HINDZINC", "APLAPOLLO"]},
    "coal": {"category": "metals", "symbols": ["COALINDIA", "NLCINDIA"]},

    # ── Tech / new-age / media ──
    "it_services_large": {"category": "tech", "symbols": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"]},
    "it_services_mid": {"category": "tech", "symbols": ["PERSISTENT", "COFORGE", "MPHASIS", "LTTS", "KPITTECH", "TATAELXSI", "CYIENT", "BSOFT", "ZENSARTECH"]},
    "new_age_internet": {"category": "tech", "symbols": ["ETERNAL", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY", "CARTRADE", "IDEAFORGE", "MAPMYINDIA"]},
    "media_entertainment": {"category": "media", "symbols": ["ZEEL", "SUNTV", "PVRINOX", "SAREGAMA", "NETWORK18", "TIPSMUSIC"]},
    "telecom": {"category": "telecom", "symbols": ["BHARTIARTL", "IDEA", "INDUSTOWER", "TATACOMM", "HFCL", "TEJASNET", "ITI"]},

    # ── Infra / construction / real estate ──
    "real_estate": {"category": "realty", "symbols": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD", "BRIGADE", "LODHA", "SOBHA", "MAHLIFE"]},
    "construction_infra": {"category": "infra", "symbols": ["LT", "KEC", "KPIL", "NCC", "KNRCON", "PNCINFRA", "HGINFRA", "GRINFRA", "IRB"]},
    "roads_highways": {"category": "infra", "symbols": ["IRB", "HGINFRA", "PNCINFRA", "KNRCON", "GRINFRA", "ASHOKA"]},

    # ── Industrials: pumps / compressors / abrasives / refractories / gases ──
    "pumps": {"category": "industrials", "symbols": ["KIRLOSENG", "SHAKTIPUMP", "KSB", "WPIL", "ROTO", "KIRLOSBROS"]},
    "compressors_gensets": {"category": "industrials", "symbols": ["CUMMINSIND", "ELGIEQUIP", "INGERRAND", "KIRLOSENG"]},
    "abrasives": {"category": "industrials", "symbols": ["CARBORUNIV", "GRINDWELL", "WENDT"]},
    "refractories": {"category": "industrials", "symbols": ["ORIENTREF", "IFGLEXPOR"]},
    "industrial_gases": {"category": "industrials", "symbols": ["LINDEINDIA"]},
    "engineering_epc": {"category": "industrials", "symbols": ["THERMAX", "ISGEC", "TRITURBINE", "HBLENGINE", "AIAENG", "ELECON"]},
    "lubricants": {"category": "materials", "symbols": ["CASTROLIND", "GULFOILLUB", "TIDEWATER"]},
    "gold_financiers": {"category": "financials", "symbols": ["MUTHOOTFIN", "MANAPPURAM", "IIFL"]},

    # ── Energy transition / water / environment ──
    "solar_value_chain": {"category": "utilities", "symbols": ["WAAREEENER", "PREMIERENE", "INSOLATION", "WEBSOL", "SOLEX", "ADANIGREEN"]},
    "wind_energy": {"category": "utilities", "symbols": ["SUZLON", "INOXWIND", "INOXGFL", "ORIENTGREEN"]},
    "water_treatment": {"category": "utilities", "symbols": ["VATECHWABAG", "IONEXCHANG", "EMSLIMITED", "WPIL"]},
    "ethanol_biofuel": {"category": "agri", "symbols": ["BALRAMCHIN", "TRIVENI", "PRAJIND", "DALMIASUG", "BAJAJHIND"]},
    "green_hydrogen_proxy": {"category": "utilities", "symbols": ["RELIANCE", "NTPC", "LINDEINDIA", "ADANIENSOL", "JSWENERGY"]},

    # ── Travel / discretionary niches ──
    "airlines": {"category": "consumer", "symbols": ["INDIGO", "SPICEJET"]},
    "travel_tourism": {"category": "consumer", "symbols": ["IRCTC", "INDIGO", "EASEMYTRIP", "THOMASCOOK", "MAHINDHOL", "BLSINTL"]},
    "packaging": {"category": "materials", "symbols": ["EPL", "UFLEX", "COSMOFIRST", "TIMETECHNO", "HUHTAMAKI", "POLYPLEX", "JINDALPOLY"]},
    "wires_electricals_d2c": {"category": "electricals", "symbols": ["POLYCAB", "KEI", "HAVELLS", "VGUARD", "RRKABEL", "ORIENTELEC"]},

    # ── Tech / new-age niches ──
    "semiconductor_proxy": {"category": "tech", "symbols": ["KAYNES", "CGPOWER", "ASMTEC", "MOSCHIP", "RIR"]},
    "defense_electronics": {"category": "defense", "symbols": ["BEL", "DATAPATTNS", "ASTRAMICRO", "PARAS", "ZENTEC", "MTARTECH"]},
    "fintech_brokers": {"category": "financials", "symbols": ["ANGELONE", "MOTILALOFS", "NUVAMA", "IIFL", "360ONE", "ANANDRATHI"]},

    # ── Materials niches ──
    "specialty_steel_pipes": {"category": "metals", "symbols": ["APLAPOLLO", "RATNAMANI", "WELCORP", "MAHSEAMLES", "JTLIND"]},
    "aluminium": {"category": "metals", "symbols": ["HINDALCO", "NATIONALUM", "VEDL"]},
    "diamond_jewellery_retail": {"category": "consumer", "symbols": ["TITAN", "KALYANKJIL", "SENCO", "THANGAMAYL", "GOLDIAM"]},
}
