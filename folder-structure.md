equipment_rental_agent/
│
├── app.py                          # 🖥️ MAIN: Streamlit Chatbot UI (Person C)
├── agent.py                        # 🧠 CORE: Parser, Scoring, Decision Logic (Person B)
├── data_loader.py                  # 💾 DATA: Loads CSVs and logs decisions (Person A & D)
├── utils.py                        # 🛠️ HELPERS: Logging, date formatting (Person D)
├── requirements.txt                # 📦 Dependencies
├── README.md                       # 📄 Project Overview
│
└── data/                           # 📊 Dataset Folder (Person A)
    ├── equipment.csv
    ├── contractors.csv
    ├── inquiries.csv               # (Optional - for testing dropdown, but chatbot uses internal)
    └── decision_log.csv            # 🔄 AUTO-GENERATED: Tracks every agent decision