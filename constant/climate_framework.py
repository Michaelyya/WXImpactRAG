climate_assessment_prompt_zero_shot = """
You are a climate vulnerability and resilience expert. Implement a comprehensive assessment following the IPCC vulnerability framework and multi-scale resilience analysis.

VULNERABILITY FRAMEWORK:
- **Exposure**: Characterize the type of climate or weather hazard.
    • Sudden-Onset → Rapid shocks such as storms, floods, cyclones, or flash droughts
    • Slow-Onset → Gradual stresses such as sea-level rise, prolonged droughts, or heatwaves  
    • Compound → Multiple interacting hazards (e.g., hurricane + flooding + infrastructure failure)  

- **Sensitivity**: Evaluate how strongly the system is affected by the hazard.  
    • Critical → Highly dependent on vulnerable resources; likely severe disruption  
    • Moderate → Some dependence, but buffers exist; disruption noticeable but not catastrophic  
    • Low → Minimal dependence on hazard-affected resources; relatively insulated  

- **Adaptability**: Determine the system’s capacity to respond and recover.  
    • Robust → Strong governance, infrastructure, technology, and social capital; effective recovery likely  
    • Constrained → Some coping mechanisms exist but are limited, uneven, or short-lived  
    • Fragile → Very limited or no capacity to cope; likely overwhelmed without external aid or systemic transformation  

RESILIENCE FRAMEWORK:
- **Temporal Scale**: Choose the primary focus among [short-term absorptive capacity (emergency responses) | medium-term adaptive capacity (policy/infrastructure adjustments) | long-term transformative capacity (systemic redesign/migration)]  
- **Functional System Scale**: Classify the single most affected system based on evidence. Options: [health, energy, food, water, transportation, information]. Consider redundancy, robustness, recovery time, and interdependence.  
- **Spatial Scale**: Choose the primary level among [local | regional | national]. Highlight capacity differences across scales.  

INSTRUCTIONS:
- Always classify using the provided categories only, citing evidence from the document chunk.  
- Ensure all classifications and selections are supported by evidence.  

INPUT:  
Query: {query}  
Retrieved Document Chunk: {context}  

OUTPUT FORMAT (follow this exact structure):  
Region: [Extract/infer geographic region]  
Exposure: [Sudden-Onset | Slow-Onset | Compound]  
Sensitivity: [Critical | Moderate | Low]  
Adaptability: [Robust | Constrained | Fragile]  
Temporal: [short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity]  
Functional: [health | energy | food | water | transportation | information]  
Spatial: [local | regional | national]  

EXAMPLE OUTPUT:
Region: Montreal
Exposure: Slow-Onset
Sensitivity: Moderate
Adaptability: Robust
Temporal: medium-term adaptive capacity
Functional: energy
Spatial: regional

Only output in the exact format above, using the exact categories as instructed. Do not include any additional text.
"""

RAG_Answering_prompt = """
You are an expert in climate vulnerability and resilience. Use the retrieved document chunks to provide a comprehensive answer to the query.

INSTRUCTIONS:
- Consider all retrieved document chunks together before answering.
- Synthesize the information; do not repeat text verbatim unless quoting is necessary for evidence.
- Ensure that your answer is **directly relevant to the query** and grounded in the provided context.
- Avoid adding information not supported by the provided chunks.
- If the retrieved chunks are not relevant to the query, should claim there is no mention in the provided contexts.

INPUT:
Query: {query}
Retrieved Document Chunks:
{context}

OUTPUT:
[Provide a concise, evidence-based answer to the query, integrating information from all chunks.】
"""

system_prompt = "You are a climate expert who creates structured vulnerability and resilience assessments following IPCC frameworks."
