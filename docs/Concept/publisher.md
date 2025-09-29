Publisher is a concept that allows you to track internal changes during the execution of graps, I will keep publishing the events.

This is mainly publisher and subscriber pattern, where you can have a single or multiple subscribers to a single publisher, and get the events as they are published, and then do whatever you want with them., you can log them, or push to Kafka, RabitMQ or any other message queue.

Currently we have ConsolePublisher, which prints the events to the console.