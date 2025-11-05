class Concept:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

class Relation:
    def __init__(self, name, from_concept, to_concept):
        self.name = name
        self.from_concept = from_concept
        self.to_concept = to_concept

    def __str__(self):
        return f"[{self.from_concept}] -({self.name})-> [{self.to_concept}]"

# Example usage
dog = Concept("Dog")
animal = Concept("Animal")
bark = Concept("Bark")

r1 = Relation("is_a", dog, animal)
r2 = Relation("can", dog, bark)

# Print the relations
print(r1)
print(r2)
