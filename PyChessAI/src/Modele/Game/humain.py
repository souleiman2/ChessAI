from Modele.Game.joueur import Joueur
from Modele.Elements.pion import Pion
import easygui_qt


# Cela est l'équivalent du joueur humain qui va pouvoir jouer (cela ne sera utile que pour la classe Game)
class Humain(Joueur):
    # le constructeur
    def __init__(self, couleurBlanc):
        super().__init__(couleurBlanc)

    # demander à l'utilisateur qu'elle pièce prendre la place du pion qui est en train d'avoir une _promotion
    def  get_promotion(self):

        options = ['Reine', 'Tour', 'Fou', 'Cavalier']
        msg = 'Choisissez la _promotion'
        titre = 'Promotion'
        choix = easygui_qt.get_choice(msg, titre, options)

        if choix == options[0]:
            return Pion.getChoices()[0]
        elif choix == options[1]:
            return Pion.getChoices()[1]
        elif choix == options[2]:
            return Pion.getChoices()[2]
        elif choix == options[3]:
            return Pion.getChoices()[3]
