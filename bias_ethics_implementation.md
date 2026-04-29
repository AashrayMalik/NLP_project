# Bias and Ethics Evaluation Implementation

This document outlines how representational bias and ethical concerns within the `r/jobs` dataset are defined and calculated for the Streamlit dashboard's "Bias and ethics notes" section. 

Given the organic, crowd-sourced nature of Reddit, the dataset inevitably captures discussions of systemic bias, discrimination, and identity-based workplace challenges. The implementation uses a lightweight, probe-based approach to surface and quantify these discussions.

## 1. How Bias is Defined

We define "bias" in this context as identity-sensitive workplace or hiring disparities reported by users. Rather than attempting complex zero-shot classification on the entire dataset, we define four distinct **Bias Probes**. Each probe is tied to a specific area of concern and mapped to a set of sub-string keyword patterns:

1. **Gender, pregnancy, and parenthood**
   *   **Definition:** Do posts describe gendered workplace treatment, pregnancy, maternity, or parenthood penalties?
   *   **Patterns:** `pregnan`, `maternity`, `female`, `woman`, `mother`, `parenthood`
2. **Race, nationality, and visa status**
   *   **Definition:** Do posts connect hiring outcomes to race, nationality, immigration status, or visa constraints?
   *   **Patterns:** `race`, `black`, `asian`, `immigrant`, `visa`, `nationality`
3. **Age and seniority**
   *   **Definition:** Do users report ageism, being too old, or age-coded rejection?
   *   **Patterns:** `ageism`, `older worker`, `too old`, `over 50`, `over 40`, `age discrimination`
4. **Disability, neurodiversity, and health**
   *   **Definition:** Do posts discuss disability, ADHD, autism, mental health, or accommodation risk?
   *   **Patterns:** `disab`, `adhd`, `autis`, `mental health`, `accommodation`, `chronic illness`

## 2. How the Calculations are Made

The calculations are performed natively in SQLite via the `load_bias_probe_summary()` function in `app.py`. The process avoids heavy NLP processing by relying on efficient substring matching over the post content.

### Step 2a: Establishing the Baseline
The system first calculates the total number of posts in the database:
```sql
SELECT COUNT(*) AS n FROM posts
```

### Step 2b: Executing the Probes
For each of the four probes, the system dynamically constructs a `WHERE` clause that chains the defined keyword patterns using `OR` and `LIKE`. It searches across both the post's title and body:
```sql
SELECT COUNT(*) AS n 
FROM posts 
WHERE lower(coalesce(title, '') || ' ' || coalesce(body, '')) LIKE '%pattern1%' 
   OR lower(coalesce(title, '') || ' ' || coalesce(body, '')) LIKE '%pattern2%' ...
```

### Step 2c: Calculating Corpus Share
The absolute match count for each probe is then divided by the `total_posts` to determine its **Share of corpus**, providing a quantifiable metric of how prevalent these bias-related discussions are relative to general job market chatter.

### Step 2d: Extracting Qualitative Evidence
To ensure the keyword matching is surfacing relevant context, the system executes a follow-up query for each probe to retrieve the top 3 highest-scoring posts (`ORDER BY score DESC LIMIT 3`) that matched the patterns. The title, score, flair, and a 260-character snippet of the body are extracted and passed to the frontend to act as concrete evidence of the community's lived experiences with these biases.
