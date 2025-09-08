def hello():
    yield "Hello"


def main():
    yield "HI"
    yield from hello()


if __name__ == "__main__":
    for msg in main():
        print(msg)
