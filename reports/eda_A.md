# Workstream A — Labels and cell identity (corrected tables, eda_shared v0.2.0)

FAFB 139,255 neurons; MaleCNS brain part 144,414. Every number is in reports/eda_A_numbers.json. The pre-correction figures are kept in figures/eda/v0.1.1/.

| Item | Finding |
| --- | --- |
| A1 Label coverage | FAFB: cell_type 98.9%, cell_class 77.2%, super_class 100%, nt_type_corrected 99.9%. MaleCNS: cell_type 99.1%, cell_class 12.6%, flywire_type 97.7%, nt_type_corrected 91.3%, dimorphism 1.6%. cell_class cannot link the two datasets. |
| A2 Where superclasses sit | Optic lobe holds 67.3% of FAFB and 68.4% of MaleCNS neurons. 86% of FAFB sensory neurons sit in sensory-periphery neuropils. GNG is mixed: sensory 34%, central 23%, ascending 19%, descending 13%. |
| A3 Type sizes | FAFB 8,846 types, 4,632 with 1–2 neurons, 149 with > 100 (127 optic); largest R1-6 (8,452). MaleCNS 8,638 types, 4,488 with 1–2, 166 with > 100 (135 optic). |
| A4 Types across superclasses | FAFB 13 types span two superclasses, none three; MaleCNS 1. |
| A5 Left and right copies | 95.2% (FAFB) and 95.3% (MaleCNS) of neurons sit in left/right pairs. 23.5% of FAFB types have unequal counts (optic 51.0%, central 21.7%); unpaired: unequal counts 4,997, no type 1,541, side unknown 29, centre 90. |
| A6 Cross-dataset matching | 7,783 of 8,846 FAFB types have a MaleCNS match, covering 95.0% of neurons: sensory periphery 98.5%, higher-order 87.7%, motor periphery 80.9%. 2,749 MaleCNS neurons list several FAFB types. |
| A7 Sex-difference labels | Labelled neurons: FAFB 1,176, MaleCNS 2,299. Share of each group: FAFB higher-order 2.3%, motor 1.85%, sensory 0.31%; MaleCNS higher-order 4.5%, motor 3.77%, sensory 0.5%. Unmatched FAFB types: higher-order 13.1%, motor 11.5%, sensory 7.3%. |
| Surprise | FAFB sensory_ascending neurons are almost never in a matched type (2.3%), while every other afferent class is above 74%. |

D18 proposal (unchanged): a neuron's homologue is the same cell type on the other side; types are paired up to min(left, right), and unpaired neurons stay in the fold of their type.
