from injectq import InjectQ


data = {"key1": "value1", "key2": "value2", "key3": "value3"}


if __name__ == "__main__":
    injector = InjectQ()
    injector.bind_factory("data_store", lambda x: data[x])

    result = injector.call_factory("data_store", "key2")
    print(result)
