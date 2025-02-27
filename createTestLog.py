import random

# 定义日志级别
log_levels = ["INFO", "WARN", "ERROR"]

# 定义异常类型
exception_types = ["IOException", "NullPointerException", "TimeoutException"]

number = 2500
# 生成日志
for i in range(number):
    log_level = random.choice(log_levels)
    timestamp = f"2025-02-27 {random.randint(0, 23):02}:{random.randint(0, 59):02}:{random.randint(0, 59):02}"
    if log_level == "ERROR":
        exception_type = random.choice(exception_types)
        error_message = f"An error occurred: {exception_type}"
        print(f"{timestamp} {log_level} - {error_message}")
    else:
        operation = random.choice(["Data ingestion", "Data processing", "Model training"])
        print(f"{timestamp} {log_level} - {operation} is in progress.")
