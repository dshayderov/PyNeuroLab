from pyn_utils import read_json, write_json, normalize, Timer


if __name__ == "__main__":
    data = {
        "name": "Peter",
        "age": 32
    }
    write_json("data.json", data)

    with Timer("Нормализация"):
        normalize([1,2,3], scale=7)