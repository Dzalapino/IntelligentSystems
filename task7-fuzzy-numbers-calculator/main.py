import matplotlib.pyplot as plt


UNARY_OPERATIONS = ["negate", "reciprocal", "absolute"]
BINARY_OPERATIONS = ["add", "subtract", "multiply", "divide"]


class TriangularFuzzyNumber:
    def __init__(self, a1, a2, a3):
        """
        Initialize a triangular fuzzy number
        :param a1: Lower bound of the fuzzy number
        :param a2: Peak of the fuzzy number
        :param a3: Upper bound of the fuzzy number
        """
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3


    def __repr__(self):
        return f"TFN: ({self.a1}, {self.a2}, {self.a3})"


    def negate(self):
        return TriangularFuzzyNumber(
            -self.a3,
            -self.a2,
            -self.a1
        )


    def reciprocal(self):
        if self.a1 == 0 or self.a2 == 0 or self.a3 == 0:
            raise ValueError("Cannot take reciprocal of a fuzzy number with 0 in its membership function.")
        return TriangularFuzzyNumber(
            1 / self.a3,
            1 / self.a2,
            1 / self.a1
        )


    def absolute(self):
        return TriangularFuzzyNumber(
            abs(self.a1),
            abs(self.a2),
            abs(self.a3)
        )


    def add(self, other_fn):
        return TriangularFuzzyNumber(
            self.a1 + other_fn.a1,
            self.a2 + other_fn.a2,
            self.a3 + other_fn.a3
        )


    def subtract(self, other_fn):
        return TriangularFuzzyNumber(
            self.a1 - other_fn.a1,
            self.a2 - other_fn.a2,
            self.a3 - other_fn.a3
        )


    def multiply(self, other_fn):
        return TriangularFuzzyNumber(
            self.a1 * other_fn.a1,
            self.a2 * other_fn.a2,
            self.a3 * other_fn.a3
        )


    def divide(self, other):
        if other.a1 == 0 or other.a2 == 0 or other.a3 == 0:
            raise ValueError("Cannot divide by a fuzzy number with 0 in its membership function.")
        return TriangularFuzzyNumber(
            self.a1 / other.a1,
            self.a2 / other.a2,
            self.a3 / other.a3
        )


    def plot(self, label="Fuzzy Number"):
        plt.plot([self.a1, self.a2, self.a3], [0, 1, 0], label=label)
        plt.fill_between([self.a1, self.a2, self.a3], [0, 1, 0], alpha=0.1)


def create_triangular_fuzzy_number():
    return TriangularFuzzyNumber(
        float(input("Enter the lower bound of the fuzzy number: ")),
        float(input("Enter the peak of the fuzzy number: ")),
        float(input("Enter the upper bound of the fuzzy number: "))
    )


def main():
    while True:
        print('negate, reciprocal, absolute, add, subtract, multiply, divide, exit? ')
        operation = input().strip().lower()
        if operation == "exit":
            break

        elif operation in UNARY_OPERATIONS:
            fuzzy_number = create_triangular_fuzzy_number()
            if operation == "negate":
                result = fuzzy_number.negate()
            elif operation == "reciprocal":
                result = fuzzy_number.reciprocal()
            elif operation == "absolute":
                result = fuzzy_number.absolute()
            print("Result:", result)

            # Plotting the fuzzy number
            fuzzy_number.plot(label="Original Fuzzy Number")
            result.plot(label="Resulting Fuzzy Number")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("Membership Degree")
            plt.title("Fuzzy Numbers")
            plt.show()

        elif operation in BINARY_OPERATIONS:
            print("First fuzzy number:")
            fuzzy_number1 = create_triangular_fuzzy_number()
            print("Second fuzzy number: ")
            fuzzy_number2 = create_triangular_fuzzy_number()

            if operation == "add":
                result = fuzzy_number1.add(fuzzy_number2)
            elif operation == "subtract":
                result = fuzzy_number1.subtract(fuzzy_number2)
            elif operation == "multiply":
                result = fuzzy_number1.multiply(fuzzy_number2)
            elif operation == "divide":
                result = fuzzy_number1.divide

            print("Result:", result)

            # Plotting the fuzzy numbers
            fuzzy_number1.plot(label="First fuzzy Number")
            fuzzy_number2.plot(label="Second fuzzy Number")
            result.plot(label="Resulting Fuzzy Number")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("Membership Degree")
            plt.title("Fuzzy Numbers")
            plt.show()

        else:
            print("Invalid operation")
            continue

if __name__ == "__main__":
    main()

