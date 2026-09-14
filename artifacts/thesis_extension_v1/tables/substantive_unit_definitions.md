| variable_or_pattern | group | substantive_unit | operational_definition | measurement_unit | documentation_status | frozen_encoded_column_count |
| --- | --- | --- | --- | --- | --- | --- |
| level | basic | Basic final-board state | player level | numeric level | features.py implementation; external Riot semantic definition not documented | 2 |
| composition_size | basic | Basic final-board state | units on final board | count | features.py/data dictionary | 2 |
| unit_presence__<unit> | units | Units/champions | named unit on final board | binary indicator | features.py | 134 |
| unit_star__<unit> | units | Units/champions | star level of named unit; zero if absent | integer star level encoded numeric | features.py; external upper-bound semantics undocumented | 134 |
| trait_presence__<trait> | traits | Traits | named trait present | binary indicator | features.py | 84 |
| trait_strength__<trait> | traits | Traits | tier_current, falling back to style | numeric Riot payload tier/style | features.py; substantive tier scale undocumented | 84 |
| item_count__<item> | items | Equipped items | occurrences of named item | count | features.py | 175 |
| structure__unique_units | structure | Composition structure | distinct named units | count | features.py | 9 |
| structure__mean_star | structure | Composition structure | mean star level across final units | mean star level | features.py | 9 |
| structure__max_star | structure | Composition structure | maximum star level | star level | features.py | 9 |
| structure__two_star_units | structure | Composition structure | units at exactly two stars | count | features.py | 9 |
| structure__three_star_units | structure | Composition structure | units at three or more stars | count | features.py | 9 |
| structure__total_items | structure | Composition structure | all equipped item occurrences | count | features.py | 9 |
| structure__unique_items | structure | Composition structure | distinct equipped item names | count | features.py | 9 |
| structure__active_traits | structure | Composition structure | traits with tier_current > 0 | count | features.py | 9 |
| structure__trait_strength_total | structure | Composition structure | sum of tier_current among active traits | sum of payload tiers | features.py; cross-trait comparability undocumented | 9 |
