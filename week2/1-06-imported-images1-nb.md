---
jupytext:
  cell_metadata_json: true
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++

pour réaliser ce TP localement sur votre ordi, {download}`commencez par télécharger le zip<./ARTEFACTS-images.zip>`

+++

# TP images (1/2)

merci à Wikipedia et à stackoverflow

```{admonition} disclaimer
:class: danger

**vous n'allez pas faire ici de traitement d'image - on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe: on le voit)**
```

+++

```{admonition} **Notions intervenant dans ce TP**
:class: tip

* création, indexation, slicing, modification  de `numpy.ndarray`
* affichage d'image (RBG, RGB-A, niveaux de gris)
* lecture de fichier `jpg`
* les autres notions utilisées sont rappelées (très succinctement)

**N'oubliez pas d'utiliser le help en cas de problème.**
```

+++

## import des librairies

+++

1. Importez la librairie `numpy`

1. Importez la librairie `matplotlib.pyplot`  
ou toute autre librairie d'affichage que vous aimez et/ou savez utiliser: `seaborn` ...

```{code-cell} ipython3
#1 
import numpy as np
#2 
import matplotlib.pyplot as plt 
```

2. optionnel - changez la taille par défaut des figures matplotlib
   par exemple choisissez d'afficher les figures dans un carré de 4x4 (en théorie ce sont des inches)

   ````{tip}
   il y a plein de façons de le faire, google et/ou stackoverflow sont vos amis...
   ````

+++

## création d'une image de couleur

````{admonition} Rappels (rapides)
:class: tip


* dans une image en couleur, les pixels sont représentés par leurs *dosages* dans les 3 couleurs primaires: `red`, `green`, `blue` (RGB)  
  ```{image} media/synthese-additive.png
  :width: 100px
  :align: right
  ```
* l'affichage à l'écran, d'une image couleur `rgb`, utilise les règles de la synthèse additive  
`(r, g, b) = (255, 255, 255)` donne la couleur blanche  
`(r, g, b) = (0, 0, 0)` donne du noir  
`(r, g, b) = (255, 0, 0)` donne du rouge
`(r, g, b) = (255, 255, 0)` donne du jaune ...
* pour afficher le tableau `im` comme une image, utilisez: `plt.imshow(im)`
* pour afficher plusieurs images dans une même cellule de notebook faire `plt.show()` après chaque `plt.imshow(...)`
````

+++

**Exercice**

1. Créez un tableau **non initialisé**, pour représenter une image carrée **de 91 pixels de côté**, d'entiers 8 bits non-signés, et affichez-le  
   ```{admonition} indices
   :class: tip
   * il vous faut pouvoir stocker 3 `uint8` par pixel pour ranger les 3 couleurs
   * on s'intéresse uniquement à la taille, et pas au contenu puisqu'on a dit "non initialisé"; que vous ayez du blanc, du noir ou du bruit, c'est OK
   ```

```{code-cell} ipython3
im = np.arange(91*91*3).reshape((91, 91, 3))
plt.imshow(im)
```

2. Transformez le en tableau blanc (en un seul slicing) et affichez-le

```{code-cell} ipython3
im[:, :] = [255, 255, 255]
plt.imshow(im)
```

3. Transformez le en tableau vert (en un seul slicing) et affichez-le

```{code-cell} ipython3
im[:, :] = [0, 255, 0]
plt.imshow(im)
```

4. Affichez les valeurs RGB du premier pixel de l'image, et du dernier

```{code-cell} ipython3
print(im[0, 0])
print(im[-1, -1])
```

5. Faites un quadrillage d'une ligne bleue, toutes les 10 lignes et colonnes et affichez-le

```{code-cell} ipython3
im[::10, ::] = [0, 0, 255]
im[::, ::10] = [0, 0, 255]
plt.imshow(im)
```

## lecture d'une image en couleur

+++

1. Avec la fonction `plt.imread` lisez le fichier `data/les-mines.jpg`

```{code-cell} ipython3
plt.imread("data/les-mines.jpg")
```

2. Vérifiez si l'objet est modifiable avec `im.flags.writeable`; si il ne l'est pas, copiez l'image

```{code-cell} ipython3
im.flags.writeable
```

3. Affichez l'image

```{code-cell} ipython3
img = plt.imread("data/les-mines.jpg")
plt.imshow(img)
```

4. Quel est le type de l'objet créé ?

```{code-cell} ipython3
img.dtype
```

5. Quelle est la dimension de l'image ?

+++

6. Quelle est la taille de l'image en hauteur et largeur ?

```{code-cell} ipython3
print(img.shape)
#533 pixels de longueur et 800 de largeur 
```

7. Quel est le nombre d'octets utilisé par pixel ?

+++

8. Quel est le type des pixels ?  
(deux types pour les pixels: entiers non-signés 8 bits ou flottants sur 64 bits)

```{code-cell} ipython3
img.dtype
#ce sont des enters non-signés sur 8 bits
```

9. Quelles sont ses valeurs maximale et minimale des pixels ?

```{code-cell} ipython3
max = np.max(img)
min = np.min(img)
print(max, min)
```

10. Affichez le rectangle de 10 x 10 pixels en haut de l'image

```{code-cell} ipython3
plt.imshow(img[:10, :10])
```

## accès à des parties d'image

+++

1. Relire l'image

```{code-cell} ipython3
plt.imread("data/les-mines.jpg")
```

2. Slicer et afficher l'image en ne gardant qu'une ligne et qu'une colonne sur 2, 5, 10 et 20  
(ne dupliquez pas le code)

```{admonition} indices
:class: tip

* vous pouvez créer plusieurs figures depuis une seule cellule  
  pour cela, faites plusieurs fois la séquence `plt.imshow(...); plt.show()`
* vous pouvez ensuite choisir de 'replier' ou non la zone *output* en hauteur;  
  c'est-à-dire d'afficher soit toute la hauteur, soit une zone de taille fixe avec une scrollbar pour naviguer  
  pour cela cliquez dans la marge gauche de la zone *output*
```

```{code-cell} ipython3
img = plt.imread("data/les-mines.jpg")

im2 = img[::2, ::]
im2 = im2[::, ::2]
plt.imshow(im2) ; plt.show()

im5 = img[::5, :]
im5 = im5[:, ::5]
plt.imshow(im5) ; plt.show()

im10 = img[::10, ::]
im10 = im10[::, ::10]
plt.imshow(im10) ; plt.show()

im20 = img[::20, ::]
im20 = im20[::, ::20]
plt.imshow(im20) ; plt.show()
```

3. Isoler le rectangle de `l` lignes et `c` colonnes en milieu d'image  
affichez-le pour `(l, c) = (10, 20)`) puis `(l, c) = (100, 200)`

```{code-cell} ipython3
#initialisation des données
l = 10 #ou 100 
c = 20  #ou 200

lig, col = img.shape[0], img.shape[1]
mil_lig = lig // 2
mil_col = col // 2

im_new = img [ (mil_lig - l//2) : (mil_lig + l//2), (mil_col - c//2) : (mil_col + c//2)]
plt.imshow(im_new)
```

## canaux RGB de l'image

+++

1. Relire l'image

```{code-cell} ipython3
plt.imread("data/les-mines.jpg")
```

2. Découpez l'image en ses trois canaux Red, Green et Blue
   (Il s'agit donc de construire trois tableaux de dimension 2)

```{code-cell} ipython3
img = plt.imread("data/les-mines.jpg")

R = img[::, ::, 0]
G = img[::, ::, 1]
B = img[::, ::, 2]

print(R, G, B)
```

3. Afficher chaque canal avec `plt.imshow`; la couleur est-elle la couleur attendue ?  
    Si oui très bien, si non que se passe-t-il ?

    ```{admonition} **rappel** table des couleurs
    :class: tip

    * `RGB` représente directement l'encodage de la couleur du pixel, et non un indice dans une table
    * donc pour afficher des pixel avec les 3 valeurs RGB pas besoin de tables de couleurs, on a la couleur
    * mais pour afficher une image unidimensionnelle contenant des nombres de `0` à `255`, 
      il faut bien lui dire à quoi correspondent les valeurs  
      (lors de l'affichage, le `255` des rouges n'est pas le même `255` des verts)
    * du coup, voyez le paramètre `cmap=` de `plt.imshow`; et notamment avec `'Reds'`,  `'Greens'` ou  `'Blues'`
    ```

```{code-cell} ipython3
plt.imshow(R); plt.show()
plt.imshow(G); plt.show()
plt.imshow(B); plt.show()

#on obtient effecitvement ni une image exclusivement rouge, ni une exclusiement verte, ... car quand on utilise imshow avec un 
#tableau à 2 dimension le cofficient de la matrice correspond a l'intensité du pixel
```

4. Corrigez vos affichages si besoin

```{code-cell} ipython3
plt.imshow(R, cmap = 'Reds' ); plt.show()
plt.imshow(G, cmap = 'Greens' ); plt.show()
plt.imshow(B, cmap = 'Blues'); plt.show()

#en effet cette fois ci on a trois image correspondant aux couleurs recherchées
```

5. Copiez l'image, et dans la copie, remplacer le carré de taille `(200, 200)` en bas à droite:
   * d'abord par un carré de couleur RGB `(219, 112, 147)` (vous obtenez quelle couleur)  
   * puis par un carré blanc avec des rayures horizontales rouges de 1 pixel d'épaisseur

```{code-cell} ipython3
#code pour un carré de couleur RGB (219, 112, 147)
cut_long = 200
cut_larg = 200

imc = img.copy()
long, larg = imc.shape[0], imc.shape[1]
imc[long - cut_long :,  larg - cut_larg:] = [219, 112, 147]
plt.imshow(imc); plt.show()

#code pour créer un carré lanc avec des rayres horiontales rouges d'un pixel

imc2 = img.copy()
long, larg = imc2.shape[0], imc2.shape[1]
imc2[long - cut_long :: 2 ,  larg - cut_larg:] = [255, 255, 255]
imc2[long - cut_long + 1 :: 2 ,  larg - cut_larg:,] = [219, 112, 147]
plt.imshow(imc2); plt.show()
```

6. enfin pour vérifier, affichez les 20 dernières lignes et colonnes du carré à rayures

```{code-cell} ipython3
print( imc2[long - cut_long : ,  larg - cut_larg:] ) 
#c'est ok on obient une ligne de blanc puis de rouge, ....
```

## transparence des images

+++

````{admonition} rappel: la transparence
**rappel** RGB-A

* on peut indiquer, dans une quatrième valeur des pixels, leur transparence
* ce 4-ème canal s'appelle le canal alpha
* les valeurs vont de `0` pour transparent à `255` pour opaque
````

+++

1. Relire l'image initiale (sans la copier)

```{code-cell} ipython3
img = plt.imread("data/les-mines.jpg")
```

2. Créez un tableau vide de la même hauteur et largeur que l'image, du type de l'image initiale, mais avec un quatrième canal

```{code-cell} ipython3
long, larg = img.shape[0], img.shape[1]

empty = np.arange(long*larg*4).reshape((long, larg, 4)) 
```

3. Copiez-y l'image initiale, mettez le quatrième canal à `128` et affichez l'image

```{code-cell} ipython3
empty[::, ::, :3] = img 
empty[::, ::, 3] = 128 

plt.imshow(empty)
```

## image en niveaux de gris en `float`

+++

1. Relire l'image `data/les-mines.jpg`

```{code-cell} ipython3
img = plt.imread("data/les-mines.jpg")
```

2. Passez ses valeurs en flottants entre 0 et 1 et affichez-la

```{code-cell} ipython3
imc = img.astype(float).copy()  #conversion en float car l'image est codé de sorte à le type des valeur soit des flotant
imc[:, :] = np.random.rand(3)

print(imc)
plt.imshow(imc)
```

3. Transformer l'image en deux images en niveaux de gris :  
a. en mettant pour chaque pixel la moyenne de ses valeurs R, G, B  
b. en utilisant la correction `Y` (qui corrige le constrate) basée sur la formule  
   `Y = 0.299 * R + 0.587 * G + 0.114 * B`  
c. optionnel: si vous pensez à plusieurs façons de faire la question a., utilisez `%%timeit` pour les benchmarker et choisir la plus rapide

```{code-cell} ipython3
#a.

imc2 = img.astype(float).copy()
moy = ( img[:,:,0] + img[:,:,1] + img[:,:,2] ) / 3
imc2[:, :, 0] = moy
imc2[:, :, 1] = moy
imc2[:, :, 2] = moy
plt.imshow(imc2); plt.show()

#b.

imc3 = img.astype(float).copy()
gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2] 
imc3[:, :, 0] = gray
imc3[:, :, 1] = gray
imc3[:, :, 2] = gray
plt.imshow(imc3) ; plt.show()
```

5. Pareil, mais cette fois utilisez la racine carrée; quel effet cette fois ?

```{code-cell} ipython3
imc4 = img.astype(float).copy()

imc4[:, :, 0] = np.sqrt(img[:,:,0])
imc4[:, :, 1] = np.sqrt(img[:,:,1])
imc4[:, :, 2] = np.sqrt(img[:,:,2])

plt.imshow(imc4); plt.show()
```

6. Convertissez l'image (de 3.a toujours) en type entier, et affichez la

```{code-cell} ipython3
imc5 = imc2.astype(np.uint8)
plt.imshow(imc5); plt.show()
```

## affichage grille de figures

+++

`````{admonition} Mettre plusieurs figures dans une grille

Mettons l'exercice sur pause pour l'instant, et voyons comment avec matplotlib, on peut mettre faire une figure composite, i.e. qui contienne plusieurs figures disposés dans une grille

**0) on dessine toujours la même chose**, juste avec des couleurs différentes

```python
# pas important, juste un exemple de truc à dessiner

X = np.linspace(0, 2*np.pi, 50)
Y = np.sin(X)
```

**1) on créé une figure globale et des sous-figures**

les sous-figures sont appelées `axes` par convention `matplotlib`  
on construit notre grille ici de 2 lignes et 3 colonnes

```python
# ici axes va être un tableau numpy de shape .. (2, 3)
fig, axes = plt.subplots(2, 3)
```

les cases pour les sous-figures sont ici dans la variable `axes`  
qui est un `numpy.ndarray` de taille 2 lignes et 3 colonnes

**2) on affiche des sous-figure dans des cases de la grille**

```python
# en haut à gauche
axes[0, 0].plot(X, Y, 'b')
axes[0, 1].plot(X, Y, 'r')
# en haut à droite
axes[0, 2].plot(X, Y, 'y')
axes[1, 0].plot(X, Y, 'k')
axes[1, 1].plot(X, Y, 'g')
axes[1, 2].plot(X, Y, 'm')
```

````{admonition} 3) cosmétique (optionnel)
:class: dropdown tip
on peut faire un peu de cosmétique, mais je vous mets en garde: 
quand on commence on ne s'arrête plus et on perd beaucoup de temps; 
préférez au début des affichages minimalistes à peu près lisibles
```python
fig.suptitle("sinus en couleur", fontsize=20) # titre général
axes[0, 0].set_title('sinus bleu')            # titre d'une sous-figure
axes[0, 2].set_xlabel('de 0 à 2 pi')          # label des abscisses
axes[1, 1].set_ylabel('de -1 à 1')            # label d'ordonnées
axes[1, 2].set_title('sinus magenta')
plt.tight_layout()                            # ajustement automatique des paddings
```
````
`````

```{code-cell} ipython3
# ce qui nous donne, mis bout à bout
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2*np.pi, 50)
Y = np.sin(X)

# le code
fig, axes = plt.subplots(2, 3)
print(f"{type(axes)=}")
print(f"{axes.shape=}")

axes[0, 0].plot(X, Y, 'b')
# axes[0, 1].plot(X, Y, 'r')
axes[0, 2].plot(X, Y, 'y')
axes[1, 0].plot(X, Y, 'k')
axes[1, 1].plot(X, Y, 'g')
axes[1, 2].plot(X, Y, 'm')

fig.suptitle("sinus en couleur", fontsize=20)
axes[0, 0].set_title('sinus bleu')
axes[0, 2].set_xlabel('de 0 à 2 pi')
axes[1, 1].set_ylabel('de -1 à 1')
axes[1, 2].set_title('sinus magenta')
plt.tight_layout();
```

## reprenons le TP

+++

**exercice**

Reprenez les trois images en niveau de gris que vous aviez produites ci-dessus:  
  A: celle obtenue avec la moyenne des rgb  
  B: celle obtenue avec la correction Y  
  C: celle obtenue avec la racine carrée

1. Affichez les trois images côte à côte
   ```text
   A B C
   ```

```{code-cell} ipython3
#Initialisation des images 
A = imc2
B = imc3
C = imc4

#Affichage des images
fig, axis = plt.subplots(1, 3)
axis[0].imshow(A) 
axis[1].imshow(B)
axis[2].imshow(C) 
```

2. Affichez-les en damier:
   ```text
   A B C
   C A B
   B C A
   ```

```{code-cell} ipython3
fig, axis = plt.subplots(3, 3)

axis[0, 0].imshow(A); axis[1, 1].imshow(A); axis[2, 2].imshow(A) 
axis[0, 1].imshow(B); axis[1, 2].imshow(B); axis[2, 0].imshow(B) 
axis[0, 2].imshow(C); axis[1, 0].imshow(C); axis[2, 1].imshow(C)

```

```{code-cell} ipython3

```
