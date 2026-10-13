"""Common French words (accents stripped, lowercase) - one signal, with
French-only accented letters, for telling a French word the learner typed
from an English one they dropped in (see app.translate.lemmatizer)."""

KNOWN_FRENCH_WORDS: set[str] = set(
    """
    le la les un une des du de au aux et ou mais donc car ni que qui quoi dont ou ce cet cette ces
    je tu il elle on nous vous ils elles me te se moi toi lui leur eux y en ne pas plus jamais rien
    mon ma mes ton ta tes son sa ses notre nos votre vos leurs
    etre avoir faire aller dire pouvoir vouloir savoir voir venir devoir prendre donner parler aimer
    manger boire dormir lire ecrire partir sortir mettre croire penser trouver passer rester sembler
    suis es est sommes etes sont ai as avons avez ont vais vas va allons allez vont fais fait faisons
    faites font peux peut pouvons pouvez peuvent veux veut voulons voulez veulent sais sait savons
    dois doit vois voit viens vient prends prend aime aimes aimons aimez aiment parle parles parlons
    bonjour bonsoir salut merci oui non bien tres beaucoup peu trop aussi encore deja toujours souvent
    aujourd hui demain hier maintenant ici la bas avec sans pour par sur sous dans chez entre vers
    avant apres pendant depuis comme comment pourquoi quand combien quel quelle quels quelles
    bon bonne mauvais grand grande petit petite nouveau nouvelle vieux vieille beau belle joli jolie
    jour nuit matin soir semaine mois annee temps fois vie monde homme femme enfant ami amie famille
    maison ville pays travail ecole livre eau pain cafe the vin chose gens personne femme mari
    chien chat voiture rue mer plage montagne soleil pluie neige film musique sport cuisine
    tout tous toute toutes autre meme chaque quelque plusieurs certain si alors puis enfin voila
    c d j l m n s t qu
    """.split()
)
