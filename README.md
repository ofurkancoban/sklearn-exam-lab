# sklearn.lab: Scikit-Learn Exam Lab

[![unique visitors](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fabacus.jasoncameron.dev%2Fget%2Fofurkancoban-sklearn-exam-lab%2Funique-visitors&query=%24.value&label=unique%20visitors&color=2b55c9&cacheSeconds=300)](https://ofurkancoban.github.io/sklearn-exam-lab/)

An interactive study site for the course script **"Machine Learning with Scikit-Learn: A Little Mathematics"** (Oliver Kramer). Every method from the script has a 3D model you can rotate and change, the key formulas, the scikit-learn code with its real output, typical exam traps and practice questions.

**Live site:** https://ofurkancoban.github.io/sklearn-exam-lab/

**Exam script (PDF):** [assets/ML_with_sklearn_script.pdf](assets/ML_with_sklearn_script.pdf)

## What is inside

Every 3D model comes with a **step-by-step tour**: the camera flies to the relevant part of the scene, the settings change for each step, key points are labeled in 3D, and a short explanation with live numbers sits above the scene. Step through with Next / Back, jump with the progress bar, or use Auto-play. "Explore freely" hands the controls back to you.

| Section | Interactive model |
| --- | --- |
| The basic idea | Iris in 3D, supervised vs. unsupervised view, X / y notation table |
| K-nearest neighbors | Query flower with its k neighbors in 3D, decision regions for every k |
| Support vector machines | Decision function f(x) as a 3D landscape with margin lines and support vectors (SMO solver in the browser), kernel "lift" demo |
| Random forests | One floating layer per tree, averaged probabilities q̄(c \| x), bootstrap sampling demo, tree diagram |
| Scaling and checking predictions | Raw vs. standardized feature space, data leakage, live confusion matrix with threshold |
| Linear regression | Fit a plane by hand, residuals and squared errors, MAE / RMSE / R² |
| Lasso | Objective surface over two weights, real Diabetes coefficient path |
| Polynomial regression | (x, x², y) space where the fit is a flat plane, under- and overfitting |
| Multilayer perceptron | ReLU neuron as a folded plane, MLP trained live on Iris in 3D |
| DBSCAN | Core / border / noise points, ε-neighborhoods, cluster growth animation, 3D chain |
| PCA | Center, find directions, project: animated in 3D, Iris PCA |
| MDS | SMACOF iterations building a 2D map, stress and Shepard plot |
| PCA before classification | Wine pipeline, GridSearchCV results per number of components |
| Choosing a model setting | K-fold cross-validation in 3D, grid search over k |
| Quick reference | Method table, formula sheet (printable), five exercises with solutions |
| Beyond the script | Optional topics from the long lecture notes: entropy, softmax, min-max scaling, OLS and gradient descent, MAPE, Wilcoxon test |
| Exam practice | Spaced-repetition review, filtered practice quiz, 30-minute mock exam, calculation drills, flashcards, progress dashboard |

## Accuracy

All datasets (Iris, Wine, Diabetes, Two Moons) were exported from scikit-learn, and every code output shown on the site comes from running the script's code with scikit-learn 1.9. The in-browser algorithms reproduce scikit-learn's results, for example DBSCAN on the script's moons sample (19 / 2 / 1 clusters for eps = 0.1 / 0.25 / 0.5) and PCA's explained variance on Iris (73.0% / 22.9%).

## Run locally

No build step. Serve the folder with any static server:

```bash
python3 -m http.server 8765
```

Then open http://localhost:8765. An internet connection is needed for three.js, MathJax and Google Fonts (loaded from CDNs).

## Keyboard shortcuts

| Key | Action |
| --- | --- |
| `/` or `Ctrl` + `K` | Search all sections |
| `←` `→` or `[` `]` | Previous / next section |
| `m` | Mark the section as studied |
| `g` | Go to Exam practice |
| `?` | Show shortcuts |

## Project structure

```
assets/               the short exam script (PDF)
index.html            page shell, navigation, script includes
css/style.css         design tokens (light and dark), layout, components
js/core.js            helpers: Three.js stage, 2D plots, UI controls, math
js/data.js            datasets and precomputed scikit-learn results
js/app.js             router, navigation, search, shortcuts, theme
js/chapters/*.js      one file per section (content + interactive models)
js/quiz.js            concept questions and calculation drills
js/questions2.js      scenario, code-reading and extra questions
js/practice.js        quiz engine, spaced repetition, mock exam, progress
js/exercises.js       exercise solutions with real outputs
```

## Visitor counter

The badge above and the number in the site's sidebar show **unique visitors of the live site**, stored in an [Abacus](https://jasoncameron.dev/abacus/) counter. The site increments it once per browser (a flag in localStorage marks a browser as counted) and only reads it on later visits; the README badge only reads it. A visitor using a new device, a private window or cleared site data is counted again, so the number is an approximation. No personal data is sent.

## Tech

Plain HTML, CSS and JavaScript, [three.js](https://threejs.org/) r147 for 3D, [MathJax](https://www.mathjax.org/) for formulas. Progress and quiz history are stored in the browser's localStorage only.

## Note

This is an unofficial study aid. The short exam script by Oliver Kramer is included in `assets/` for reference; all rights to it remain with its author. The long version of the lecture notes is not included.
