def read_req():
    file_path = "requirements.txt"

    with open(file_path, mode="r") as file:
        lines = file.readlines()

    lines = [line.strip("\n ") for line in lines if line.strip("\n ") != ""]

    return lines
