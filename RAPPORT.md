# Rapport de projet : détecteur de phishing par apprentissage automatique

## 1. Objectif du projet

Le but de ce projet est de classer un email dans deux catégories : soit c'est un
email de phishing (une arnaque), soit c'est un email légitime (un vrai email
normal). En plus de donner une réponse, le programme donne aussi un score de
risque entre 0 et 100 %, et quelques explications pour dire pourquoi il pense que
l'email est dangereux ou pas.

La priorité que j'ai choisie est le rappel (recall) sur la classe phishing.
Autrement dit, il vaut mieux lever une fausse alerte de temps en temps que de
laisser passer un vrai email dangereux. Dans un contexte de sécurité, rater un
phishing est plus grave que de se tromper sur un email normal.

## 2. Les données utilisées

Pour entraîner un modèle, il faut beaucoup d'exemples. J'ai récupéré plusieurs
jeux de données publics qui contiennent des emails déjà étiquetés (phishing ou
légitime) :

- Le dataset `ealvaradob/phishing-dataset` sur Hugging Face.
- Le dataset `naserabdullahalam/phishing-email-dataset` sur Kaggle.
- Plusieurs corpus classiques utilisés en recherche : Enron, SpamAssassin,
  Nazario, Nigerian_Fraud, Ling, CEAS_08.

Tous ces emails ont été remis dans un format unique : un fichier CSV avec deux
colonnes, `text` (le contenu de l'email) et `label` (0 = légitime, 1 = phishing).

Pour le nettoyage, j'ai fait attention à un point important : je n'ai pas
supprimé les liens (URLs) du texte. Au début je pensais les enlever, mais en
fait les liens sont un signal très utile pour repérer le phishing, donc il faut
les garder. Par contre j'enlève les en-têtes techniques de l'email (From:, Date:,
MIME-Version:, etc.) qui ne servent à rien pour la classification.

Au final le jeu de données nettoyé contient environ 92 000 emails après tous les
ajouts décrits plus bas.

## 3. Première approche : un modèle simple

Pour commencer, j'ai voulu un modèle simple et solide, sans aller trop vite vers
quelque chose de compliqué. J'ai donc utilisé :

- **TF-IDF** pour transformer le texte en chiffres. Le TF-IDF regarde quels mots
  apparaissent dans l'email et donne plus d'importance aux mots rares et
  significatifs. J'ai pris les mots seuls et les groupes de deux mots (1 et 2
  grammes).
- **Régression logistique** comme modèle de classification. C'est un modèle
  classique, rapide, et qui marche bien sur du texte.

J'ai aussi mis l'option `class_weight='balanced'` parce que les deux classes
n'ont pas exactement le même nombre d'exemples, et ça aide le modèle à ne pas
trop favoriser la classe majoritaire.

Pour évaluer le modèle correctement, j'ai séparé les données en deux : une partie
pour l'entraînement et une partie pour le test (split stratifié). Comme ça les
résultats sont mesurés sur des emails que le modèle n'a jamais vus.

Ce premier modèle marchait déjà très bien sur les chiffres globaux (environ 99 %
de bonnes réponses). Mais ces chiffres cachaient un vrai problème.

## 4. Premier problème rencontré : trop de faux positifs

En testant le modèle à la main avec des emails légitimes, je me suis rendu compte
qu'il se trompait souvent dans un cas précis : les vrais emails qui contiennent un
lien, ou qui parlent d'argent, de paiement, ou de banque, étaient classés comme
phishing alors qu'ils étaient parfaitement normaux.

Le problème vient des données. Dans le corpus, presque tous les emails qui
parlent de "relevé bancaire" ou de "compte en ligne" sont des phishing (environ
91 %). Du coup le modèle a appris une mauvaise règle : "si l'email parle de
banque, c'est du phishing". Ce n'est pas vrai dans la vraie vie, parce que les
banques envoient aussi plein de vrais emails.

C'est un point que j'ai trouvé intéressant : un modèle peut avoir un excellent
score global et quand même apprendre des raccourcis faux à cause des données.

## 5. Deuxième approche : une cascade en plusieurs étapes

Pour régler ce problème, au lieu d'avoir un seul modèle qui décide tout seul,
j'ai construit une cascade avec plusieurs étapes :

1. **Étape 1 — pré-filtre.** Le modèle TF-IDF + régression logistique sert de
   premier filtre large. J'ai abaissé son seuil à 0.35 pour qu'il laisse passer
   le maximum de phishing possible (quitte à avoir des fausses alertes à ce
   stade). Si le score est vraiment très bas (en dessous de 0.20), on décide tout
   de suite que c'est légitime sans aller plus loin.
2. **Étape 2 — features structurées.** Ici on extrait des informations précises
   sur l'email : des features sur les URLs (voir partie 6) et des features sur le
   texte (ton urgent, demande de mot de passe, etc.).
3. **Étape 3 — méta-classifieur.** Un deuxième modèle (régression logistique)
   prend toutes ces informations et donne le score final. Le verdict est :
   phishing si le score dépasse 0.60, suspect entre 0.35 et 0.60, sinon légitime.
4. **Étape 4 — explications.** Une couche de règles transparentes qui sert
   uniquement à expliquer la décision, jamais à la changer. Par exemple, elle ne
   dit jamais "il y a un lien donc c'est du phishing" toute seule.

Un point important sur lequel j'ai fait attention : éviter la fuite de données
(data leakage). Toutes les étapes utilisent la même séparation train/test (même
graine aléatoire 42). Le classifieur d'URL s'entraîne sur la partie train, et le
méta-classifieur s'entraîne seulement sur la partie test du modèle de base. Comme
ça, les scores qu'il apprend sont de vraies prédictions sur des données jamais
vues, et pas des chiffres "trichés".

## 6. Les features d'URL

Plutôt que de traiter le lien comme du simple texte, j'ai écrit du code pour
analyser la structure de l'URL, un peu comme le ferait un humain. Les features
que j'extrais sont par exemple :

- L'extension du domaine (TLD) : certaines extensions comme `.tk`, `.ml`, `.ga`
  sont plus souvent utilisées par les arnaques.
- La présence d'une adresse IP directe à la place d'un nom de domaine.
- La profondeur des sous-domaines (un domaine avec beaucoup de points est
  suspect).
- La présence de mots comme `login`, `verify`, `secure`, `password` dans le
  chemin de l'URL.
- La ressemblance avec un vrai domaine connu (typosquatting), par exemple
  `amaz0n.tk` qui imite `amazon`.
- La présence de paramètres de redirection (`redirect=`, `url=`, etc.).

J'ai utilisé un RandomForest pour ce sous-modèle d'URL.

Problème rencontré ici : le score AUC de ce classifieur d'URL est resté autour de
0.88, et je n'ai pas réussi à le monter beaucoup plus haut. Après analyse, ce
n'est pas vraiment un défaut de réglage, c'est une limite des données. Dans le
corpus réel, les liens des phishing ressemblent souvent à des liens normaux (ce
sont de vieux spams), donc le signal "structure de l'URL" est faible au départ.
Le test qui compte vraiment (comparer `amazon.fr` qui doit être sûr et `amaz0n.tk`
qui doit être dangereux) fonctionne bien, donc j'ai gardé ce sous-modèle même
avec un AUC pas parfait. De toute façon ce n'est qu'une information parmi
d'autres pour le méta-classifieur.

## 7. Corriger les données plutôt que tricher

Une chose que j'ai voulu éviter, c'est de "bidouiller" les poids du modèle à la
main pour avoir de beaux chiffres. À la place, j'ai préféré corriger la cause du
problème, c'est-à-dire les trous dans les données. J'ai ajouté des exemples
générés automatiquement, mais de façon variée et réaliste :

- Des emails de phishing avec de vraies URLs dangereuses (faux domaines,
  extensions suspectes, adresses IP) pour que le sous-modèle d'URL apprenne les
  bons signaux.
- Des vrais emails de banque légitimes (relevés, notifications de compte) avec de
  vrais noms de domaine de banque, pour casser la fausse règle "banque =
  phishing".

Après ces ajouts, un vrai relevé bancaire qui était classé à 99.9 % de risque est
redescendu à 3.2 %, alors qu'un phishing bancaire avec un faux domaine reste à
90-100 %. C'est exactement le comportement voulu.

## 8. Deuxième gros problème : les emails en français

Quand j'ai testé le modèle avec mes propres emails en français, j'ai eu une très
mauvaise surprise : presque tous mes emails normaux étaient classés à 90-100 % de
phishing. Même une phrase totalement banale comme "réunion demain 14h" était
détectée comme dangereuse.

Après réflexion, la cause n'était pas les URLs mais la langue. Tous les datasets
et tous mes ajouts étaient en anglais. Donc pour le modèle, du texte en français
c'est "inconnu", et il a tendance à mettre les choses inconnues dans la classe
phishing. La même phrase en anglais était bien classée à 1 %, mais en français à
99 %.

Pour corriger ça, j'ai généré des emails synthétiques en français dans les deux
classes en même temps :

- Des emails légitimes français (messages personnels, confirmations, banque, avec
  ou sans lien).
- Des emails de phishing français (urgence, demande d'identifiants, liens
  dangereux).

Le point important est de mettre des exemples français dans les deux classes,
sinon le modèle apprendrait juste "français = légitime", ce qui serait une
nouvelle mauvaise règle.

Résultat : les vrais emails français descendent maintenant à 0-2 % de risque (au
lieu de 90-100 %), et les phishing français restent à 98-100 %. Les emails en
anglais n'ont pas changé.

## 9. Résultats finaux

Voici les principaux chiffres obtenus à la fin (mesurés sur la partie test, donc
sur des emails jamais vus pendant l'entraînement).

Modèle de base (étape 1) :

| Métrique | Valeur |
|----------|--------|
| Accuracy | 0.99 |
| Précision (phishing) | 0.99 |
| Rappel (phishing) | 0.99 |
| F1 (phishing) | 0.99 |

Méta-classifieur (décision finale) :

| Métrique | Valeur |
|----------|--------|
| AUC | 0.999 |
| Précision (phishing) | 0.99 |
| Rappel (phishing) | 0.99 |

Sous-modèle d'URL (RandomForest) :

| Métrique | Valeur |
|----------|--------|
| AUC | 0.88 |
| Précision (phishing) | 0.80 |
| Rappel (phishing) | 0.84 |

J'ai aussi entraîné, à titre de comparaison, un modèle plus avancé basé sur
DistilBERT (un transformer). Il donne des résultats très proches du modèle simple
(environ 0.98 de F1), mais il est beaucoup plus lourd et lent, surtout sans carte
graphique. J'ai donc gardé le modèle simple comme modèle principal, et DistilBERT
reste optionnel.

## 10. Limites du projet

- Les données sont surtout en anglais. Même après mes ajouts en français, le
  modèle reste meilleur en anglais.
- Le modèle se base sur les mots. Il peut donc être trompé par des attaques
  nouvelles ou des formulations inhabituelles.
- Il reste des faux positifs et des faux négatifs. Aucune détection n'est
  parfaite.
- Un cas reste difficile : un email ultra court et générique, sans expéditeur,
  sans marque et sans lien (par exemple "merci pour votre commande") est ambigu
  même pour un humain, et le modèle le met dans "suspect" (environ 55 %).

C'est pour ça que ce projet est un outil éducatif. Il ne remplace pas une vraie
protection email professionnelle (filtrage, authentification SPF/DKIM/DMARC,
analyse des pièces jointes, etc.).

## 11. Ce que j'ai appris

Le point le plus marquant de ce projet, c'est que la qualité des données compte
souvent plus que le choix du modèle. Mes deux plus gros problèmes (les faux
positifs sur la banque, et les emails français) venaient tous les deux des
données, pas de l'algorithme. À chaque fois, la bonne solution était de corriger
les données plutôt que de bricoler le modèle.

J'ai aussi appris à ne pas faire confiance à un seul chiffre global : un modèle
à 99 % d'accuracy peut quand même avoir un gros défaut caché qu'on ne voit qu'en
testant à la main avec des cas concrets.
