# Github Repository zum _What-if_ Simulationstool

Streamlit Tool zur Masterarbeit:
> Einsatz von Machine Learning zur Prädiktion der Kindersterblichkeitsrate und Identifikation relevanter Einflussfaktoren: Eine simulationsbasierte Analyse auf Länderebene

[Git Repo child mortlity rate prediction](https://github.com/HamzahFayad/master_thesis_child_mortality)

### Projekt

Ein szenariobasiertes Simulationstool, welches die U5MR Vorhersage auf Länderebene hypothetisch simuliert

#### Home
- Einführung

#### Demo
- Auswahl Land und Jahre aus Referenzzeitraum
- interaktive Feature Slider: relative prozentuale Änderungen der Features 
##### Base Vorhersage:
- Base: Vorhersagen als Quantiles (0.25, 0.5, 0.75) zur Erkennung der Unsicherheit in den Vorhersagen; 
Fokus Quantil anhand globalem Referenzvergleich
- SHAP Plots zur globalen und lokalen Erklärbarkeit der Vorhersagen  
##### simulierte Vorhersage
- Unsicherheiten als Quantiles (0.25, 0.5, 0.75); Best-/Worst-Case Vorhersagen; inklusive absolute & prozentuale Differenzen
- SHAP Plot zur lokalen Erklärbarkeit der simulierten Vorhersage pro Jahr  
##### Sensitivität
- Sensitivitätsvorschau nach Einkommensklassen / Weltregionen je Featureänderung