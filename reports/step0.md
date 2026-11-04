# Step 0 — corrected tables (eda_shared v0.2.0, config v0.2.0)

## FAFB

- Neurons: **139,255** = 139,241 kept + 14 added; 7 dropped
- super_class: optic 77,537; central 32,383; sensory 16,904; visual_projection 8,038; ascending 1,750; descending 1,303; sensory_ascending 612; visual_centrifugal 524; motor 110; endocrine 80; unknown 14
- Edges: τ=1 15,091,983; ≥5 **2,700,513**
- Region: 78 values (reports/step0_fafb_region_counts.csv); ties 232; unknown 281
- Self-connections dropped: 0
- Central population (is_optic False, known region): 45,203
- Transmitter correction: known_nt 77,169 rows, 18,513 changed; predicted 62,086 rows, 0 changed
- Dopamine: 5,909 → 638

## MaleCNS (brain part)

- Neurons kept: **144,414** of 165,122 Traced; dropped 20,708; ascending kept 1,841; sensory_ascending kept 534
- Brain edges: τ=1 21,706,010; ≥5 **5,143,528**
- Synapses: brain 99,982,458; nerve cord 24,041,864; no location 251
- Region ties 76; region changed by the brain rule 1,657
- Self-connections dropped: 101
- Central population (is_optic False, known region): 45,642
- super_class: optic 89,388; central 32,160; visual_projection 9,202; sensory 8,918; ascending 1,841; descending 1,312; visual_centrifugal 563; sensory_ascending 534; unknown 302; motor 107; endocrine 71; efferent 16
- Transmitter correction: kc_override 4,064 rows, 4,064 changed; predicted 140,350 rows, 0 changed
