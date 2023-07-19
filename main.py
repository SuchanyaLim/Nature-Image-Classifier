import naive
import fuzzy


if __name__ == "__main__":

    input_filepath = input("Enter image filepath: ")

    while input_filepath != "":

        print("Probabilities: [tundra, forest, desert, ocean]")

        print("----------------------------------------------------------")
        print("NAIVE_BAYES_CLASSIFIER RESULT: This picture is most likely " + naive.classifier(input_filepath)[0])
        print(naive.classifier(input_filepath)[1])
        print("----------------------------------------------------------")
        print("FUZZY_CLASSIFIER RESULT: This picture is most likely " + fuzzy.classifier(input_filepath)[0])
        print(fuzzy.classifier(input_filepath)[1])
        print("----------------------------------------------------------")

        input_filepath = input("Enter image filepath: ")
