"""Notebook-spezifische Software- und Parametererklärungen."""

GUIDES = {
    "01": r"""
## Software und einstellbare Größen

Dieses Notebook führt in den Datenzugriff ein. Die eigentliche Logik liegt in
dataset_pipeline.py, damit dieselben Regeln später nicht versehentlich anders
implementiert werden.

**Verwendete Python-Werkzeuge**

- pathlib verwaltet Pfade unabhängig vom Startordner des Notebooks.
- urllib lädt die öffentliche MAT-Datei bei Bedarf von Zenodo.
- hashlib vergleicht die MD5-Prüfsumme und erkennt beschädigte Downloads.
- h5py liest die MATLAB-v7.3/HDF5-Datei, ohne MATLAB zu benötigen.
- NumPy stellt die Messzyklen als Arrays der Form (Zyklen, Sensoren, Samples)
  bereit und schreibt die kompakten NPZ-Exporte.
- scikit-learn liefert GroupShuffleSplit. Dadurch bleibt jede UGM-ID vollständig
  in genau einem Split.

**Wichtige Eingaben**

- transform = "stored" nutzt die auf Zenodo gespeicherten logarithmischen
  Sensorwerte; transform = "log1p" wendet zusätzlich NumPy log1p an.
- RANDOM_STATE legt die reproduzierbare Gruppenauswahl fest.
- BASE_TEST_SIZE und VALIDATION_SIZE_OF_DEVELOPMENT steuern die Anteile.
- PHYSICAL_SENSOR und SUB_SENSOR_INDEX bestimmen den Kanal. Für dieses Seminar
  bleiben sie absichtlich auf sensorA und 0; die Sensorachse muss Länge 1 haben.
- BASE_SEGMENT_MAX_UGM = 500 und EXTRA_TEST_MIN_UGM = 501 trennen den normalen
  Entwicklungsbereich vom exklusiven letzten Abschnitt.

Nach einer Änderung an Split-Konstanten müssen alle nachfolgenden Notebooks neu
ausgeführt werden. Test und test_extra dürfen niemals für Punktwahl,
Korrelation oder Hyperparameteroptimierung benutzt werden.
""",
    "02": r"""
## Software, Bedienung und Plotparameter

Oben werden nur GAS und TRANSFORM verändert. Danach sollte **Run All** verwendet
werden, damit jede Grafik dieselbe Auswahl zeigt.

**Eingabeparameter**

- GAS akzeptiert einen Namen aus GAS_TARGETS, beispielsweise "acetone",
  "hydrogen" oder "ethyl_acetate".
- TRANSFORM ist "stored" oder "log1p". Die zweite Variante ist eine zusätzliche
  Transformation und nicht der ursprüngliche Widerstand.

**Verwendete Software**

- NumPy lädt NPZ-Dateien, berechnet Quantile, Korrelationen und die lineare
  Kleinste-Quadrate-Lösung.
- Matplotlib erzeugt Histogramme, ECDF, Scatterplots, Signalbänder und
  Temperaturprofile. figsize ändert die Größe, alpha die Transparenz, s die
  Punktgröße und bins die Anzahl der Histogrammklassen.
- dataset_pipeline liefert die unveränderten UGM-Splits, Zielnamen,
  Temperaturgrenzen und Exporte.

Die lineare Einpunkt-Baseline löst Ziel = a mal Sensorwert + b mit
numpy.linalg.lstsq. Sie ist kein nichtlineares Sensormodell, sondern eine
Orientierung. In SciPy könnte alternativ scipy.stats.pearsonr für Korrelationen
verwendet werden; NumPy reicht hier aus und vermeidet eine zusätzliche API.

Beim Interpretieren der Scatterplots sind Punktwolke, systematische Krümmung,
Ausreißer und unterschiedliche Konzentrationsdichte wichtiger als nur ein
einzelner Korrelationswert.
""",
    "03": r"""
## Software, Modelle und Eingabeparameter

Der zentrale Eingabeparameter ist GAS. Alle 48 zulässigen Temperaturgrenzpunkte
werden geprüft; ausgewählt wird ausschließlich anhand des Validation-RMSE des
Power Laws.

**Drei Modelle**

- Lineare Regression: C = a mal x + b.
- Linear nach log1p: C = a mal log(1 + x) + b.
- Power Law: C = A mal (x / s) hoch n. Die Skala s ist der Trainingsmedian und
  verbessert die numerische Kondition.

NumPy berechnet die linearen Parameter mit numpy.linalg.lstsq. Das Power Law
wird durch Logarithmieren ebenfalls als lineares Problem gelöst. Dafür müssen
Sensorwert und Ziel positiv sein. Für ein echtes nichtlineares Fitproblem könnte
man scipy.optimize.curve_fit oder scipy.optimize.least_squares verwenden; diese
Funktionen erlauben Startwerte, Parametergrenzen und robuste Verlustfunktionen.

**Was kann verändert werden?**

- GAS wählt das Zielgas.
- Die Liste der zulässigen Punkte kommt aus temperature_boundary_points.
- Als Auswahlmetrik könnte RMSE durch MAE ersetzt werden; dies ändert jedoch die
  Bedeutung der Optimierung.
- Negative Vorhersagen werden bewusst nicht abgeschnitten, damit Modellfehler
  sichtbar bleiben.

Test und test_extra dienen erst nach Punktwahl und Fit zur Bewertung.
""",
    "04": r"""
## Software, Modelle und einstellbare Parameter

Dieses Notebook erweitert Notebook 03 auf mehrere Messpunkte.

**Eingaben**

- GAS wählt das Zielgas.
- N_FEATURES liegt zwischen 2 und 13. Es wird höchstens ein Punkt pro
  Temperaturstufe aufgenommen, damit nicht viele fast identische Punkte
  derselben Stufe dominieren.

**Methoden**

- Multivariate lineare Regression verwendet numpy.linalg.lstsq.
- Die log1p-Variante transformiert jede Eingangsspalte vor der Regression.
- Das Mehrpunkt-Power-Law besitzt pro Merkmal einen Exponenten und wird im
  logarithmischen Raum angepasst.
- Die greedy Auswahl ergänzt in jedem Schritt denjenigen zulässigen Punkt, der
  den Validation-RMSE des Power Laws am stärksten verbessert.

NumPy übernimmt Matrixrechnung und Logarithmen, Matplotlib zeigt Auswahlpfad,
Metriken und Paritätsplots. Vergleichbare fertige Alternativen sind
sklearn.linear_model.LinearRegression und scipy.optimize.least_squares.

Eine größere Merkmalszahl ist nicht automatisch besser: N_FEATURES beeinflusst
Flexibilität, Multikollinearität und Extrapolationsrisiko. Der Auswahlpfad
benutzt Train und Validation; Test sowie test_extra bleiben unangetastet.
""",
}

GUIDES.update({
    "05": r"""
## Software, physikalische Annahmen und Parameter

**Eingaben**

- GAS wählt das zu quantifizierende Gas.
- N_ALA_SEGMENTS bestimmt die Zahl adaptiver linearer Abschnitte. Mehr Segmente
  rekonstruieren feiner, erzeugen aber zweimal so viele ALA-Merkmale.
- RIDGE_ALPHAS ist eine Folge positiver Regularisierungsstärken. None erzeugt
  logarithmisch verteilte Werte von 10 hoch -4 bis 10 hoch 6.

physics_features.py teilt den Zyklus in 24 bekannte Temperaturphasen. Pro Phase
werden Start, Ende, Hub, Anfangssteigung, tau63, Exponentialfehler und ein
Qualitätsflag berechnet. tau63 ist die Zeit bis 63,2 Prozent des beobachteten
Hubs und entspricht nur bei einer Antwort erster Ordnung der physikalischen
Zeitkonstante.

Die ALA-Stützstellen werden aus dem Trainingsmedian gelernt. Für jedes Segment
werden Mittelwert und Steigung erzeugt. NumPy übernimmt Vektorisierung und
Ausgleichsgeraden; Matplotlib visualisiert Phasenfit und ALA-Rekonstruktion.
sklearn.linear_model.Ridge löst die lineare Regression mit L2-Regularisierung.
alpha = 0 nähert sich unregularisierten kleinsten Quadraten; großes alpha
schrumpft Koeffizienten stärker.

Für einen vollständigen exponentiellen Fit könnte scipy.optimize.curve_fit mit
R_inf, Amplitude und tau als Parametern eingesetzt werden. Bei kurzen,
nichtmonotonen Phasen ist tau63 robuster und leichter zu erklären. Skalierung,
ALA-Grenzen und alpha-Auswahl sehen keine Testdaten.
""",
    "06": r"""
## Software, Versuchsaufbau und veränderbare Parameter

Dieses Notebook ist absichtlich eine Negativdemonstration und darf nicht als
gültige Baseline verwendet werden.

**Eingaben**

- GAS wählt das Ziel; Ethanol zeigt den Unterschied besonders deutlich.
- N_ALA_SEGMENTS steuert die Anzahl der ALA-Segmente.
- HELD_OUT_UGM_FRACTION legt den Anteil vollständig unbekannter UGM-IDs fest.
- RANDOM_STATE_DEMO macht Gruppenwahl, Zeilensplit und Bäume reproduzierbar.

sklearn.model_selection.GroupShuffleSplit hält zuerst komplette UGMs zurück.
train_test_split zerlegt danach den Rest absichtlich zeilenweise und erzeugt
UGM-Leakage. sklearn.ensemble.ExtraTreesRegressor kombiniert viele stark
randomisierte Entscheidungsbäume.

Wichtige Extra-Trees-Parameter:

- n_estimators: Zahl der Bäume; mehr reduziert Zufallsschwankungen, kostet Zeit.
- min_samples_leaf: Mindestzahl je Blatt; 1 erlaubt nahezu vollständiges
  Memorieren, größere Werte glätten.
- max_features: Anteil der Merkmale pro Teilung.
- n_jobs = -1 nutzt alle verfügbaren CPU-Kerne.

Die Scatterplots vergleichen Wiedererkennen bekannter UGMs mit echter
Generalisierung. Hohe Trainingsgüte allein ist kein Qualitätsnachweis.
""",
    "07": r"""
## Software, Distanzmodell und Parameter

**Eingaben**

- GAS wählt das Zielgas.
- N_ALA_SEGMENTS steuert die ALA-Basis.
- N_NOISE_FEATURES legt die Anzahl eindeutig irrelevanter Kontrollmerkmale fest.
- RANDOM_STATE_DEMO reproduziert dieses Kontrollrauschen.

NumPy erzeugt Rohwertblöcke, erste Differenzen und mit numpy.fft.rfft ein
Frequenzspektrum. sklearn.neighbors.KNeighborsRegressor sagt den Zielwert aus
den k ähnlichsten Trainingszeilen vorher. Die Option weights = "distance"
gewichtet nahe Nachbarn stärker; p = 2 entspricht euklidischer Distanz.

k wird in diesem Notebook einmal auf der kompakten ALA-Baseline aus
3, 5, 7, 11, 21 und 31 gewählt und danach für alle Dimensionen festgehalten.
Kleines k ist flexibel und rauschempfindlich, großes k glättet stärker.

Vor der Distanzberechnung wird jede Spalte mit Trainingsmittelwert und
Trainingsstandardabweichung z-standardisiert. Ohne Skalierung würden Merkmale
mit großen Zahlenwerten dominieren. scipy.spatial.distance bietet weitere
Distanzfunktionen; scikit-learn übernimmt hier Fit und Vorhersage direkt.

Die 1.000 Rauschmerkmale sind ein kontrolliertes Experiment. Sie dürfen nicht
als reale Sensorinformation interpretiert werden.
""",
    "08": r"""
## Software, Suchräume und Bedienung der FESR-Baseline

Dieses Notebook läuft standardmäßig über alle Namen in GAS_TARGETS. Zwei
Feature-Extraction- und zwei Feature-Selection-Verfahren ergeben vier
Kandidaten pro Gas.

**Feature Extraction**

- Equidistant: 120 gleich lange Segmente mit je 12 Samples; Mittelwert und
  Steigung ergeben bei einem Kanal 240 Merkmale.
- ALA: 50 anhand des Rekonstruktionsfehlers platzierte Segmente; Grenzen werden
  nur aus dem Trainingsmedian gelernt, Mittelwert und Steigung ergeben 100
  Merkmale.

**Feature Selection und Regression**

- Pearson sortiert nach absoluter Train-Korrelation zum Ziel.
- RFE-LSR aus sklearn.feature_selection.RFE passt wiederholt
  sklearn.linear_model.Ridge an und entfernt das betragsmäßig schwächste
  Merkmal.
- sklearn.cross_decomposition.PLSRegression projiziert korrelierte Merkmale auf
  wenige latente Komponenten und regressiert dort auf die Konzentration.

**Einstellbare Suchräume**

- FEATURE_COUNT_GRID enthält die getesteten Anzahlen ausgewählter Merkmale.
- PLS_COMPONENT_GRID enthält die getesteten PLS-Dimensionen. Die Komponentenzahl
  darf die Merkmalszahl nicht überschreiten.
- Ridge-alpha für RFE-LSR steht in fesr_baseline.py und ist standardmäßig 1.
- Die ALA-Segmentzahl ist im Extraktionsblock auf 50 gesetzt.

Validation wählt FE, FS, Merkmalszahl und PLS-Komponenten. Die Tabellen zeigen
auch Testwerte aller Kandidaten, diese beeinflussen die Auswahl aber nicht.
csv und json aus der Python-Standardbibliothek schreiben die Baseline-Artefakte;
NumPy, scikit-learn und Matplotlib übernehmen Berechnung und Darstellung.
""",
})
