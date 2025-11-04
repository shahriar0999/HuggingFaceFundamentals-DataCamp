# Filter the documents
filtered = wikipedia.filter(lambda row: "football" in row["text"])

# Create a sample dataset
example = filtered.select(range(1))

print(example[0]["text"])


## output:
"""
Luis Miguel Aparecido Alves (born May 25, 1985), known as Gugu, is a Brazilian football player currently playing for Iraklis Psachna F.C.
    
    External links
    
    1985 births
    Living people
    Brazilian men's footballers
    Thrasyvoulos F.C. players
    Ionikos F.C. players
    Super League Greece players
    Expatriate men's footballers in Greece
    Brazilian expatriate men's footballers
    Men's association football midfielders
"""