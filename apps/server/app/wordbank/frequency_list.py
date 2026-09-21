"""Starter frequency-ranked Spanish vocabulary used as the candidate pool for
introducing new words. Roughly ordered most- to least-common; good enough to
seed the "new word" selector without needing an external frequency corpus.
Swap this for a proper frequency dictionary later if desired."""

from app.wordbank.function_words import SPANISH_FUNCTION_WORD_LEMMAS

FREQUENCY_RANKED_ES = [
    "ser", "estar", "tener", "hacer", "poder", "decir", "ir", "ver", "dar", "saber",
    "querer", "llegar", "pasar", "deber", "poner", "parecer", "quedar", "creer", "hablar", "llevar",
    "dejar", "seguir", "encontrar", "llamar", "venir", "pensar", "salir", "volver", "tomar", "conocer",
    "vivir", "sentir", "tratar", "mirar", "contar", "empezar", "esperar", "buscar", "existir", "entrar",
    "trabajar", "escribir", "perder", "producir", "ocurrir", "entender", "pedir", "recibir", "recordar", "terminar",
    "permitir", "aparecer", "conseguir", "comenzar", "servir", "sacar", "necesitar", "mantener", "resultar", "leer",
    "caer", "cambiar", "presentar", "crear", "abrir", "considerar", "oir", "acabar", "convertir", "ganar",
    "formar", "traer", "partir", "morir", "aceptar", "realizar", "suponer", "comprender", "lograr", "explicar",
    "preguntar", "tocar", "reconocer", "estudiar", "alcanzar", "nacer", "dirigir", "correr", "utilizar", "pagar",
    "ayudar", "gustar", "jugar", "escuchar", "cumplir", "ofrecer", "descubrir", "levantar", "intentar", "usar",
    "casa", "tiempo", "año", "dia", "vida", "hombre", "mujer", "mundo", "mano", "parte",
    "ojo", "trabajo", "lugar", "agua", "noche", "vez", "nino", "cosa", "palabra", "estado",
    "pais", "ciudad", "problema", "hijo", "gente", "historia", "punto", "forma", "momento", "amigo",
    "gobierno", "libro", "familia", "grupo", "razon", "cabeza", "voz", "arte", "amor", "madre",
    "padre", "sistema", "idea", "ley", "guerra", "dinero", "trabajo", "escuela", "puerta", "coche",
    "grande", "pequeno", "bueno", "malo", "nuevo", "viejo", "mismo", "mucho", "poco", "otro",
    "todo", "alguno", "ninguno", "primero", "ultimo", "largo", "corto", "alto", "bajo", "joven",
    "feliz", "triste", "facil", "dificil", "importante", "posible", "imposible", "necesario", "claro", "seguro",
    "rojo", "azul", "verde", "amarillo", "negro", "blanco", "bonito", "feo", "rico", "pobre",
    "yo", "tu", "el", "ella", "nosotros", "vosotros", "ellos", "este", "ese", "aquel",
    "aqui", "alli", "ahora", "hoy", "manana", "ayer", "siempre", "nunca", "tambien", "solo",
    "muy", "mas", "menos", "bien", "mal", "cuando", "donde", "como", "porque", "aunque",
    "pero", "y", "o", "si", "no", "con", "sin", "para", "por", "sobre",
    "entre", "hasta", "desde", "durante", "segun", "comer", "beber", "dormir", "caminar", "correr",
    "nadar", "cocinar", "estudiar", "aprender", "ensenar", "comprar", "vender", "viajar", "bailar", "cantar",
    "familia", "hermano", "hermana", "abuelo", "abuela", "tio", "tia", "primo", "esposo", "esposa",
    "perro", "gato", "pajaro", "pez", "arbol", "flor", "sol", "luna", "estrella", "cielo",
    "mar", "montana", "rio", "playa", "bosque", "ciudad", "pueblo", "calle", "plaza", "mercado",
]

# Candidate pool for the "new word" selector: content words only. A few
# entries here are ambiguous between a content and a function reading in
# isolation (e.g. "bajo" as the adjective "short" vs. the preposition
# "under") - excluded anyway, since there's no context here to disambiguate
# and losing one adjective candidate from an otherwise large pool is cheap
# insurance against forcing the wrong sense in.
NEW_WORD_CANDIDATES_ES = [w for w in FREQUENCY_RANKED_ES if w not in SPANISH_FUNCTION_WORD_LEMMAS]
