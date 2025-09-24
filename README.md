# Task 4.4 — CLI CSV Analyzer

This repository contains a **command-line CSV Analyzer tool**, developed in Python and deployed using Docker.  
The application demonstrates how data processing utilities can be packaged into containers for easy distribution and execution.  

The tool provides:
- 📊 **CSV summaries** — rows, columns, headers, and memory usage.  
- 🔍 **Per-column profiling**:
  - Numeric → mean, std, quartiles, min, max  
  - Datetime → min/max auto-detected  
  - Text → unique count + top-N frequent values  
- 📜 **Run history** — optional SQLite logging of past runs.  
- 📂 **File flexibility** — analyze CSV files mounted into the container or stream data from stdin.

---

## 📂 Code Overview
- **cli_csv_analyzer.py**  
  Main Python script containing:
  - CSV analysis logic using `pandas` and `numpy`.  
  - Per-column profiling functions.  
  - Command-line interface built with `argparse`.  
  - Optional run history stored in `SQLite`.  

- **Dockerfile**  
  Defines how the CLI tool is packaged into a Docker image.  

---

## ▶️ Running the App Locally (without Docker)

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Eshitamjn20/SWE40006_Task4.4.git
   cd SWE40006_Task4.4

2. ** Set up a Python environment **
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. ** Install requirements **
   ```bash
   pip install -r requirements.txt
   ```     
4. ** Run the app **
   ```bash
   python app.py
   ```
5. Visit http://localhost:4000 in your browser to see it running

## 🐳Pulling from Docker Hub
1. ** Build the image: **
   ```bash
    docker build -t eshita20/csv-analyzer:1.0 .
   ```
2. ** Push to Docker Hub: **
 ```bash
    docker login
    docker push eshita20/csv-analyzer:1.0
```
3. ** Run the analyzer (mounting the current folder into /mnt in the container): **
 ```bash
    docker run --rm -v ${PWD}:/mnt eshita20/csv-analyzer:1.0 analyze /mnt/your_file.csv # /mnt is the container so ensure the file is uploaded into it
```
4. ** View history **
 ```bash
docker run --rm -v csvan_data:/data eshita20/csv-analyzer:1.0 history
```



## 👩‍💻 Author  

**Eshita Mahajan (104748964)**  
SWE40006 – Software Deployment and Evolution (Semester II, 2025)  

   


   
   
