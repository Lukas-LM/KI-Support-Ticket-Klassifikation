# 📌 KI-Support-Ticket-Pipeline für Apache Spark

Ein produktionsreifes, modulares Backend-System, das eingehende Support-Tickets (JIRA API) vollautomatisch analysiert, thematisch über Machine Learning klassifiziert und für Business-Intelligence-Auswertungen (Power BI) in einer SQL-Datenbank strukturiert.

## 🚀 Das System-Architektur-Konzept
Das Projekt ist nach dem **Enterprise-Pipeline-Prinzip** und strenger *Separation of Concerns* aufgebaut. Es teilt sich in drei Phasen:

1. **Die Daten- & Clustering-Pipeline (`model_selection.py`):** Holt 10.000 historische JIRA-Tickets der Apache Software Foundation über die API. Ein `SentenceTransformer` generiert 384-dimensionale Text-Embeddings. Um das Rauschen der Framework-Begriffe zu eliminieren, ist eine **Principal Component Analysis (PCA)** zur Dimensionsreduktion auf 2D vorgeschaltet, bevor ein **K-Means-Modell (K=6)** das mathematische Zuordnungsgesetz festlegt.
2. **Die relationale Datenbank (Star-Schema):** Die berechneten Daten werden punkt- und rauschfrei in eine SQLite-Datenbank exportiert, aufgeteilt in eine zentrale Faktentabelle (`fact_tickets`) und eine dynamisch über einen Keyword-Filter generierte Dimensionstabelle (`dim_clusters`) für Power BI.
3. **Der automatisierte Live-Betrieb (`predict.py`):** Ein Cloud-Worker, der über **GitHub Actions** im zeitgesteuerten Intervall gestartet wird. Er holt neue Live-Tickets (ab ID 10.001) ab, stempelt sie mit der aktuellen Uhrzeit, transformiert sie über die gespeicherten PCA/Classifier-Modelle (XGBoost/Random Forest mit einem getesteten F1-Score von bis zu 1.0) und hängt sie inkrementell an die Datenbank an.

## 📊 Power BI Dashboard
Das integrierte Power BI Dashboard importiert die SQLite-Datenbank live, verknüpft Fakt- und Dimensionstabelle über eine performante 1:n-Beziehung und liefert dem Management geschäftskritische Echtzeit-KPIs (Ticket-Volumen, Dringlichkeits-Matrizen und zeitliche Trend-Analysen).

## 🛠️ Tech-Stack
* **Language:** Python 3 (Pandas, SQLAlchemy, joblib)
* **AI/ML:** scikit-learn (KMeans, PCA), SentenceTransformers, XGBoost / Random Forest
* **Cloud & DevOps:** GitHub Actions (Cron-Scheduling, CI/CD-Automation)
* **BI-Analytics:** Power BI Desktop (Relationales Daten-Modeling)
