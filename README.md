# Domain Assignment using AI

## Overview
This solution assigns domains to requirements using semantic similarity.

## Approach
- Requirements and domain descriptions are converted into embeddings using SentenceTransformers
- Cosine similarity is used to find the closest domain
- A heuristic boost is applied based on domain-specific keywords
- Confidence is calculated based on similarity score and score gap
- Ambiguity is flagged when multiple domains have similar scores

## Output
The script generates:
- domain_assignment_output table in DuckDB
- domain_assignment_output.csv file

## How to run
```bash
python domain_assignment.py
