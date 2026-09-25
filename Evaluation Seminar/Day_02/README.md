# Seminar-Neuaufbau

Die bisherigen Notebooks im Elternordner bleiben als Referenz erhalten, gehören
aber nicht zum neuen Ablauf.

## Verbindliche Datenregeln

- Öffentliche Quelle: Zenodo DOI 10.5281/zenodo.6821340.
- Es wird nur Sensor A, Kanal/Sub-Sensor 0 geladen.
- Die Sensorachse hat immer Länge 1.
- UGM 1-500 werden gruppenweise in Train, Validation und Test geteilt.
- Die Aufteilung der UGM 1-500 ist 64 % / 16 % / 20 %.
- UGM 501-906 bilden ausschließlich den zusätzlichen Split test_extra.
- Ein Zyklus derselben UGM-ID kann niemals in mehreren Splits liegen.
- Alle Konzentrationen bleiben enthalten; es gibt keinen 300-ppb-Filter mehr.
- Punktwahl und Korrelation verwenden ausschließlich Trainingsdaten.
- Zulässig sind zunächst nur rohe Einzelpunkte an Temperaturphasengrenzen.
- Jeder Export existiert mit gespeicherten Sensorwerten und mit log1p-Transformation.

Hinweis: Zenodo beschreibt den gespeicherten Kanal bereits als logarithmischen
Sensorwiderstand. Die Variante log1p ist daher eine zusätzliche mathematische
Transformation der gespeicherten Werte und keine Rekonstruktion des Widerstands.

## Notebooks und Code

Jedes Notebook enthält direkt nach der Einführung einen eigenen Abschnitt
„Software und einstellbare Größen“. Darin werden die verwendeten NumPy-,
SciPy-, scikit-learn-, Matplotlib-, h5py- und Python-Werkzeuge eingeordnet,
zulässige Eingabeparameter erklärt und Train-/Validation-/Test-Grenzen
festgehalten. Die zentrale Textquelle notebook_guides.py wird von allen
Buildern eingebettet, damit die Lehrtexte beim Neuerzeugen erhalten bleiben.

- dataset_pipeline.py: Laden, Splitten, Transformieren und Exportieren.
- 01_Datensatz_und_Einzelpunkte.ipynb: Datensatzbasis und Punktvarianten.
- 02_Daten_visualisieren.ipynb: auswählbares Gas, Targetverteilungen,
  Rohsignalbänder, alle 48 Gas-gegen-Punkt-Scatterplots sowie lineare
  Einpunkt-Baselines mit R², MSE, RMSE und MAE auf allen vier Splits.
- 03_Ein_Merkmal_Power_Law_vs_Lineare_Regression.ipynb: fairer Vergleich
  desselben Einzelpunkts mit linearer Regression, linearer Regression auf
  log1p(x) und einem empirischen Power Law.
- 04_Mehrere_Merkmale_Power_Law_vs_Linear.ipynb: mehrere Rohpunkte gemeinsam,
  greedy über Power-Law-Validierungs-RMSE gewählt, mit Vergleich gegen beide
  multivariaten linearen Varianten. Standardmäßig zeigt Wasserstoff mit allen
  13 Temperaturstufen ein funktionierendes Positivbeispiel.
- 05_Physikalische_Merkmale_Zeitkonstanten_ALA.ipynb: Zeitkonstanten und
  weitere Dynamikmerkmale für alle 24 Temperaturphasen sowie eine auf dem
  Trainingsmedian gelernte Adaptive Lineare Approximation. Ridge-Modelle
  vergleichen Rohgrenzpunkte, Phasendynamik, ALA und beide Merkmalsgruppen.
- 06_Overfitting_Zufallssplit_vs_Unbekannte_UGMs.ipynb: bewusste
  Negativdemonstration mit zeilenweisem Zufallssplit, UGM-Leakage und einem
  separaten Test aus vollständig unbekannten UGM-IDs. Ethanol dient als
  besonders klares Overfitting-Beispiel.
- 07_Curse_of_Dimensionality.ipynb: distanzgewichtetes k-NN wird schrittweise
  mit ALA, Phasendynamik, Rohsignal, Ableitungen, Spektrum und irrelevanten
  Kontrollmerkmalen überladen. Metriken, Scatterplots und Abstandskontrast
  zeigen die Verschlechterung bei wachsender Dimension.
- 08_FESR_Baseline_Alle_Gase.ipynb: verbindliche Ein-Kanal-FESR-Baseline für
  alle zehn Gase nach dem DAV³E-Prinzip. Äquidistante Segmentierung und ALA
  liefern jeweils Mittelwert und Steigung. Pearson und RFE-LSR bilden die
  beiden Feature-Selection-Kandidaten, PLSR übernimmt die Regression. Auswahl
  erfolgt mit Train/Validation, beide Tests bleiben unangetastet.
- fesr_baseline.py: wiederverwendbare Python-Implementierung der
  FESR-Segmentierung, RFE-LSR-Auswahl und PLS-Regression.
- physics_features.py: wiederverwendbare Extraktion der Phasen- und
  ALA-Merkmale für genau einen Sensorkanal.
- notebook_guides.py: notebook-spezifische Software-, Methoden- und
  Parametererklärungen für die komplette Neuauflage.
- _build_notebook.py und _build_visualization_notebook.py: reproduzierbare
  Erzeugung der ersten beiden Notebooks.
- _build_power_law_notebook.py: reproduzierbare Erzeugung von Notebook 03.
- _build_multifeature_notebook.py: reproduzierbare Erzeugung von Notebook 04.
- _build_physics_notebook.py: reproduzierbare Erzeugung von Notebook 05.
- _build_overfitting_notebook.py: reproduzierbare Erzeugung von Notebook 06.
- _build_curse_dimensionality_notebook.py: reproduzierbare Erzeugung von
  Notebook 07.
- _build_fesr_notebook.py: reproduzierbare Erzeugung von Notebook 08.

## Verbindliche FESR-Baseline

Ab Notebook 08 gilt FESR als gemeinsame klassische Baseline. Die Resultate
werden reproduzierbar unter Data/seminar_baselines gespeichert:

- fesr_all_gases_metrics.csv: R², MSE, RMSE und MAE für alle Splits;
- fesr_candidate_combinations.csv: Validation- und Testmetriken aller
  40 Kombinationen aus zehn Gasen, zwei FE- und zwei FS-Varianten;
- fesr_selected_features.csv: ausgewählte Segmentmerkmale je Gas;
- fesr_config.json: vollständige Ein-Kanal-Konfiguration und Suchräume.

## Export

Vom Repository-Hauptordner:

    .\.venv\Scripts\python.exe "Evaluation Seminar\Day_02\dataset_pipeline.py"

Die Dateien liegen unter Data/seminar_point_datasets. Pro Gas und Transformation
werden der höchstkorrelierte Punkt, der Punkt mit der größten einfachen
Selektivitätsmarge und der beste Punkt jeder Temperaturstufe exportiert.

Zusätzlich gibt es je Transformation:

- all_raw_values: alle 1.440 Werte, Form (n, 1, 1440);
- all_temperature_boundaries: alle 48 erlaubten Grenzpunkte, Form (n, 1, 48).

Alle Dateien enthalten train, val, test und test_extra sowie sämtliche Gasziele,
Feuchte, UGM-IDs und Ursprungsindizes.

## FESR als scikit-learn-Pipeline

- `09_FESR_mit_sklearn_Pipeline.ipynb`: transparenter Ein-Kanal-Ablauf mit
  Segmentmerkmalen, StandardScaler, Pearson/RFE-Auswahl und PLSRegression.
  GroupKFold und GridSearchCV lernen alle Schritte innerhalb der Trainingsfolds.
  Separate Validierung, beide Tests, Merkmalsmarkierungen und Scatterplots sind enthalten.
