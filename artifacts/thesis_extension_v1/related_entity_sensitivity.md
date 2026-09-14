# Related-entity sensitivity analysis

The contract CV groups all eight observations from a match, so match overlap is zero. However,
the ordinary five folds contain 1,009–1,052 participant identities that also occur in their
corresponding fitting fold. This can make ordinary CV optimistic if stable player-specific board
patterns are learnable, even though participant identifiers themselves are prohibited features.

Training matches form 1155 player-connected components;
the largest contains 457 matches
(13.1% of training matches). A stricter
GroupKFold sensitivity kept every connected component within one fold, achieved zero participant
overlap in all folds, and produced mean F1 0.848
(fold SD 0.007). Fold sizes vary because connected components are indivisible.
This post-hoc sensitivity does not replace the authoritative chronological split or alter selection.
