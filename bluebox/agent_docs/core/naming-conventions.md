# Routine Naming & Documentation Conventions

Routines are vectorized and stored in databases for other agents to discover via semantic search. Clear, precise metadata is **essential** — a routine with a vague name or missing description is invisible and unusable.

## Routine Name

**Format:** `snake_case` with a `verb_site_noun` pattern and **3+ segments**.

The name MUST include the site or service name so it makes sense in isolation. Another agent reading ONLY the name — with no other context — should know what site this targets and what it does.

| Good | Bad | Why |
|------|-----|-----|
| `get_premierleague_standings` | `get_standings` | Standings from where? |
| `search_premierleague_matches_by_season` | `search_matches` | Which sport? Which site? |
| `fetch_amtrak_train_schedules` | `get_data` | Completely generic |
| `download_arxiv_paper_pdf` | `download_paper` | Which paper repository? |
| `list_espn_upcoming_fixtures` | `list_fixtures` | Which sports platform? |
| `get_github_repo_stars` | `get_content_item` | Content from where? What item? |

**Rules:**
- Always start with a verb: `get_`, `search_`, `fetch_`, `list_`, `download_`, `create_`, `submit_`
- Always include the **site/service name**: `premierleague`, `amtrak`, `arxiv`, `espn`, `github`
- Include the domain noun: `standings`, `matches`, `flights`, `players`, `teams`
- Add qualifiers when needed: `_by_season`, `_one_way`, `_with_details`
- Use `snake_case` only — no camelCase, no spaces, no hyphens
- Minimum 3 underscore-separated segments: `verb_site_noun`

## Routine Description

**Minimum 8 words.** Must answer three questions:

1. **What does it do?** (the action)
2. **What inputs does it take?** (the parameters)
3. **What data does it return?** (the output structure)

### Examples

**Good (all three questions answered):**
> Fetches Premier League standings for a given competition ID and season ID, returning team names, positions, wins, draws, losses, goals scored, goals conceded, and total points.

> Searches for one-way flights from an origin airport to a destination on a specific date, returning a list of flights with airline, departure time, arrival time, duration, stops, and price.

**Bad (missing information):**
> "Get standings" — too short, missing input/output info
> "A routine for the Premier League" — doesn't say what it does or returns
> "Fetches data from the API" — which API? what data? what format?

### Template

> `{Verb}s {what} for a given {param1} and {param2}, returning {field1}, {field2}, {field3}, and {field4}.`

## Parameter Names

**Format:** `snake_case`, descriptive, never ambiguous.

| Good | Bad | Why |
|------|-----|-----|
| `competition_id` | `id` | Ambiguous — id of what? |
| `season_year` | `year` | Could mean any year |
| `departure_date` | `date` | Which date? |
| `team_name` | `name` | Name of what? |
| `search_query` | `q` | Cryptic |
| `page_number` | `page` | Acceptable but `page_number` is clearer |

## Parameter Descriptions

**Minimum 3 words.** Must explain:
1. What the value represents
2. Expected format or range (when applicable)

### Examples

**Good:**
> "The unique competition identifier, typically a numeric ID (e.g. 1 for Premier League)"
> "Departure date in YYYY-MM-DD format (e.g. 2024-12-25)"
> "Season year as a 4-digit number (e.g. 2024 for the 2024-25 season)"

**Bad:**
> "ID" — too terse, ambiguous
> "The season" — doesn't explain format
> "query" — just restates the parameter name

## Non-Obvious Parameters: Sourcing is MANDATORY

If a parameter value is NOT something a human would naturally know — opaque numeric IDs, internal slugs, encoded tokens, UUIDs — the description **MUST** explain where to get valid values. Without sourcing, the routine is unusable.

**How to identify non-obvious parameters:** names ending in `_id`, `_slug`, `_code`, `_token`, `_key`, `_hash`, or any numeric/integer parameter that represents an internal identifier.

### Examples

**Good (includes sourcing):**
> "Internal competition ID. Obtain from the get_competitions routine or the /competitions API endpoint. Example: 1 = Premier League, 2 = Championship."

> "Season ID as used by the Premier League API. Use the get_seasons routine to list valid season IDs for a competition. Example: 418 = 2023-24 season."

> "Team slug as it appears in the site URL path (e.g. 'arsenal', 'manchester-united'). Find by calling get_teams or navigating to the team page."

**Bad (no sourcing — where do I get these?):**
> "The competition ID" — which competition? where do I look it up?
> "Season identifier" — what values are valid? how do I find them?
> "Internal team code" — completely opaque, no way to discover valid values

**Rule of thumb:** if you can't google the value, the description must say how to get it.

## Why This Matters

Other agents search the routine database with natural language queries like:
- "Find me Premier League standings"
- "Search for flights from LAX to JFK"

If your routine is named `get_data` with description "fetches data", it will never match these queries. But `get_league_standings` with a rich description will rank highly and be selected for execution.
