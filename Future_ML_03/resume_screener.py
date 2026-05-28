import re
import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '  ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(text, skill_db):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skill_db]
    matcher.add("SkillPatterns", patterns)
    doc = nlp(text)
    matches = matcher(doc)
    extracted = set()
    for match_id, start, end in matches:
        extracted.add(doc[start:end].text.lower())
    return list(extracted)

class ResumeScreener:
    def __init__(self, job_description, skill_database):
        self.raw_jd = job_description
        self.clean_jd = clean_text(job_description)
        self.skill_db = [skill.lower() for skill in skill_database]
        self.jd_skills = extract_skills(self.clean_jd, self.skill_db)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def screen_candidates(self, resumes_df):
        results = []
        for idx, row in resumes_df.iterrows():
            candidate_name = row['Candidate_Name']
            raw_resume = row['Resume_Text']
            cleaned_resume = clean_text(raw_resume)
            
            resume_skills = extract_skills(cleaned_resume, self.skill_db)
            matched_skills = list(set(resume_skills) & set(self.jd_skills))
            missing_skills = list(set(self.jd_skills) - set(resume_skills))
            
            tfidf_matrix = self.vectorizer.fit_transform([self.clean_jd, cleaned_resume])
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity_score = round(float(similarity_matrix[0][0]) * 100, 2)
            
            results.append({
                "Rank": 0, 
                "Candidate Name": candidate_name,
                "Match Score (%)": similarity_score,
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
            })
            
        output_df = pd.DataFrame(results)
        output_df = output_df.sort_values(by="Match Score (%)", ascending=False).reset_index(drop=True)
        output_df['Rank'] = output_df.index + 1
        return output_df

def main():
    master_skills = [
        "Python", "Machine Learning", "SQL", "Tableau", "Java", "AWS", "NLP", 
        "Docker", "React", "Project Management", "Data Analysis", "Excel"
    ]
    
    target_job_description = """
    Looking for a Data Analyst or Machine Learning Specialist. 
    Must have solid proficiency in Python, SQL, and Data Analysis. 
    Experience building models with Machine Learning or building dashboards with Tableau is highly desired.
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "archive (4)", "Resume", "Resume.csv")
    print(f"\n📡 Loading dataset from: {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find data file at '{csv_path}'.")
        return

    raw_data = pd.read_csv(csv_path)
    
    text_col = 'Resume_str' if 'Resume_str' in raw_data.columns else raw_data.columns[1]
    id_col = 'ID' if 'ID' in raw_data.columns else raw_data.columns[0]
    
    processed_df = pd.DataFrame({
        "Candidate_Name": raw_data[id_col].astype(str),
        "Resume_Text": raw_data[text_col]
    }).head(10)  

    print("⚡ Processing production dataset rows...")
    screener = ResumeScreener(target_job_description, master_skills)
    r_df = screener.screen_candidates(processed_df)
    
    print("\n🏆 Production Screening Results Matrix:")
    print(r_df[["Rank", "Candidate Name", "Match Score (%)", "Matched Skills"]].to_string(index=False))

if __name__ == "__main__":
    main()
