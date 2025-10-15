Dutsinma Crime Prediction System — Mobile Upload Guide (Streamlit Cloud)

Files inside this ZIP:
- app.py
- dutsinma_crime_mock.csv
- model.joblib
- requirements.txt
- README.txt
- about.txt

How to deploy from your phone (Streamlit Cloud):
1. Open https://share.streamlit.io/ on your phone and sign in with GitHub.
2. Create a new GitHub repository and upload all files from this ZIP.
   - In GitHub mobile: New -> Create repository -> Add files -> Upload files -> Commit
3. In Streamlit Cloud click "New app" -> "Deploy from a repository" and connect your GitHub repo.
4. Set main file to app.py and click deploy.
5. Open the given app URL and login (sidebar):
   Username: Binnaseer1
   Password: An@25787238

Notes:
- The app uses a demo dataset and a pre-trained model for immediate demo.
- If model.joblib is a dummy (due to environment), you can train using the included train script on a desktop and re-upload the model.