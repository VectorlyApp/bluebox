# Example Routines

Example routines are in [example_routines/](example_routines/). Each has a `*_routine.json` and `*_input.json` file.

## Amtrak Train Search

**File:** `amtrak_one_way_train_search_routine.json`

Searches for train schedules on Amtrak. Demonstrates chaining fetch operations via sessionStorage and using storage placeholders.

```json
{
    "name": "amtrak_routine",
    "description": "A routine to search for trains on Amtrak",
    "incognito": true,
    "operations": [
        {"type": "navigate", "url": "https://www.amtrak.com/"},
        {"type": "sleep", "timeout_seconds": 2.0},
        {
            "type": "fetch",
            "endpoint": {
                "description": "Amtrak station/location autocomplete. GET with query parameter searchTerm; returns JSON with autoCompleterResponse.autoCompleteList.",
                "url": "https://www.amtrak.com/services/MapDataService/AutoCompleterArcgis/getResponseList?searchTerm=\"{{origin}}\"",
                "method": "GET",
                "headers": {"Accept": "application/json, text/plain, */*"},
                "body": {},
                "credentials": "same-origin"
            },
            "session_storage_key": "amtrak_autocomplete_stations_origin"
        },
        {
            "type": "fetch",
            "endpoint": {
                "description": "Amtrak station/location autocomplete. GET with query parameter searchTerm; returns JSON with autoCompleterResponse.autoCompleteList.",
                "url": "https://www.amtrak.com/services/MapDataService/AutoCompleterArcgis/getResponseList?searchTerm=\"{{destination}}\"",
                "method": "GET",
                "headers": {"Accept": "application/json, text/plain, */*"},
                "body": {},
                "credentials": "same-origin"
            },
            "session_storage_key": "amtrak_autocomplete_stations_destination"
        },
        {
            "type": "sleep",
            "timeout_seconds": 3.0
        },
        {
            "type": "fetch",
            "endpoint": {
                "description": "Look for one-way amtrak tickets",
                "url": "https://www.amtrak.com/dotcom/journey-solution-option",
                "method": "POST",
                "headers": {
                    "Content-Type": "application/json",
                    "Origin": "https://www.amtrak.com",
                    "Referer": "https://www.amtrak.com",
                    "Accept": "application/json, text/plain, */*",
                    "x-amtrak-trace-id": "{{sessionStorage:ibsession.sessionid}}"
                },
                "body": {
                    "journeyRequest": {
                        "fare": {"pricingUnit": "DOLLARS"},
                        "alternateDayOption": false,
                        "type": "OW",
                        "journeyLegRequests": [
                            {
                                "origin": {
                                    "code": "{{sessionStorage:amtrak_autocomplete_stations_origin.autoCompleterResponse.autoCompleteList.0.stationCode}}",
                                    "schedule": {"departureDateTime": "\"{{departureDate}}\"T00:00:00"}
                                },
                                "destination": {
                                    "code": "{{sessionStorage:amtrak_autocomplete_stations_destination.autoCompleterResponse.autoCompleteList.0.stationCode}}"
                                },
                                "passengers": [
                                    {"id": "P1", "type": "F", "initialType": "adult"}
                                ]
                            }
                        ],
                        "customer": {"tierStatus": "MEMBER"},
                        "isPassRider": false,
                        "isCorporateTraveller": false,
                        "tripTags": true,
                        "singleAdultFare": true,
                        "cascadesWSDOTFilter": false
                    },
                    "initialJourneyLegOnly": false,
                    "reservableAccomodationOptions": "ALL"
                },
                "credentials": "same-origin"
            },
            "session_storage_key": "amtrak_search_one_way"
        },
        {
            "type": "return",
            "session_storage_key": "amtrak_search_one_way"
        }
    ],
    "parameters": [
        {
            "name": "origin",
            "description": "The origin city or station code",
            "type": "string",
            "required": true
        },
        {
            "name": "destination",
            "description": "The destination city or station code",
            "type": "string",
            "required": true
        },
        {
            "name": "departureDate",
            "description": "The departure date",
            "type": "string",
            "required": true
        }
    ]
}
```

## Download Arxiv Paper

**File:** `download_arxive_paper_routine.json`

Downloads a PDF paper from arxiv.org. Demonstrates single download operation.

```json
{
    "name": "Download Arxiv Paper",
    "description": "Download arxive paper as a PDF",
    "operations": [
        {
            "type": "download",
            "endpoint": {
                "headers": {},
                "method": "GET",
                "credentials": "omit",
                "url": "https://arxiv.org/pdf/\"{{paper_id}}\""
            },
            "filename": "\"{{paper_id}}\".pdf"
        }
    ],
    "incognito": true,
    "parameters": [
        {
            "examples": ["1706.03762"],
            "min_length": 1,
            "name": "paper_id",
            "description": "paper id on arxiv",
            "type": "string",
            "required": true,
            "max_length": 200
        }
    ]
}
```

## Massachusetts Corp Search

**File:** `massachusetts_corp_search_routine.json`

Searches the MA Secretary of State corporate database. Demonstrates full UI automation and js_evaluate for DOM scraping.

```json
{
  "name": "massachusetts_corp_search",
  "description": "Search for corporations and business entities registered in Massachusetts using the Secretary of State's corporate database.",
  "parameters": [
    {
      "name": "entity_name",
      "type": "string",
      "required": true,
      "description": "The name of the corporation or entity to search for (e.g., 'Microsoft Corporation')"
    }
  ],
  "operations": [
    {
      "type": "navigate",
      "url": "https://corp.sec.state.ma.us/corpweb/CorpSearch/CorpSearch.aspx"
    },
    {
      "type": "sleep",
      "timeout_seconds": 3.5
    },
    {
      "type": "scroll",
      "delta_y": 100
    },
    {
      "type": "click",
      "selector": "input[name='ctl00$MainContent$txtEntityName']"
    },
    {
      "type": "input_text",
      "selector": "input[name='ctl00$MainContent$txtEntityName']",
      "text": "\"{{entity_name}}\"",
      "clear": true
    },
    {
      "type": "click",
      "selector": "select[name='ctl00$MainContent$ddRecordsPerPage']"
    },
    {
      "type": "sleep",
      "timeout_seconds": 0.75
    },
    {"type": "press", "key": "arrowdown"},
    {"type": "press", "key": "arrowdown"},
    {"type": "press", "key": "enter"},
    {
      "type": "click",
      "selector": "input[name='ctl00$MainContent$btnSearch']"
    },
    {
      "type": "wait_for_url",
      "url_regex": "CorpSearchResults\\.aspx",
      "timeout_ms": 10000
    },
    {
      "type": "sleep",
      "timeout_seconds": 1
    },
    {
      "type": "js_evaluate",
      "js": "(()=>{const table=document.getElementById('MainContent_SearchControl_grdSearchResultsEntity');if(!table){return{error:'Table not found',results:[]};}const rows=table.querySelectorAll('tr.GridRow');const results=[];for(const row of rows){const cells=row.querySelectorAll('td,th');const link=row.querySelector('a.link');const entity={entity_name:cells[0]?.textContent?.trim()||'',id_number:cells[1]?.textContent?.trim()||'',old_id_number:cells[2]?.textContent?.trim()||'',address:cells[3]?.textContent?.trim().replace(/\\s+/g,' ')||'',link:link?.href||''};results.push(entity);}return{results,count:results.length};})()",
      "session_storage_key": "corp_search_results",
      "timeout_seconds": 10
    },
    {
      "type": "return",
      "session_storage_key": "corp_search_results"
    }
  ]
}
```

## Polymarket Bets

**File:** `get_new_polymarket_bets_routine.json`

Fetches newest bets from Polymarket API. Demonstrates optional integer parameters with defaults.

```json
{
    "name": "Get New Polymarket Bets",
    "description": "Get new polymarket bets from the API",
    "operations": [
        {
            "type": "navigate",
            "url": "https://polymarket.com/"
        },
        {
            "type": "sleep",
            "timeout_seconds": 2.5
        },
        {
            "type": "fetch",
            "endpoint": {
                "headers": {
                    "Origin": "https://polymarket.com",
                    "Accept": "application/json, text/plain, */*"
                },
                "method": "GET",
                "credentials": "include",
                "url": "https://gamma-api.polymarket.com/events/pagination?limit=\"{{limit}}\"&active=true&archived=false&closed=false&order=startDate&ascending=false&offset=\"{{offset}}\"&exclude_tag_id=100639&exclude_tag_id=102169"
            },
            "session_storage_key": "newest_polymarket_events"
        },
        {
            "type": "return",
            "session_storage_key": "newest_polymarket_events"
        }
    ],
    "incognito": true,
    "parameters": [
        {
            "description": "Number of newest events (bets) to fetch; defaults to 20 if not provided.",
            "type": "integer",
            "required": false,
            "min_value": 1,
            "default": 20,
            "examples": [10, 20, 50],
            "name": "limit",
            "max_value": 100
        },
        {
            "description": "Pagination offset into the newest events list; defaults to 0 for the first page.",
            "type": "integer",
            "required": false,
            "min_value": 0,
            "default": 0,
            "examples": [0, 20, 40],
            "name": "offset"
        }
    ]
}
```
