#pragma once

#include <iostream>
#include <cstdarg>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <iterator>

class Timestamp
{
public:
    friend std::ostream& operator<<(std::ostream& os, const Timestamp&){
        auto now = std::chrono::system_clock::now();
        std::time_t t_now = std::chrono::system_clock::to_time_t(now);
        std::tm* tm = std::localtime(&t_now);

        std::cout << "[" << std::setfill('0')
        << std::setw(2) << tm->tm_hour << ":"
        << std::setw(2) << tm->tm_min << ":"
        << std::setw(2) << tm->tm_sec << "]";

        return os;
    }
};

template <typename T>
struct is_iterable {
private:
    template <typename U>
    static auto test(int) -> decltype(std::begin(std::declval<U>()), std::end(std::declval<U>()), std::true_type{});

    template <typename>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

namespace Logger {
    enum LogLevel {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
        Critical = 4
    };

    const char* levelString(LogLevel level) {
        switch (level) {
            case Debug: return "[DEBUG]";
            case Info: return "[INFO]";
            case Warning: return "[WARNING]";
            case Error: return "[ERROR]";
            case Critical: return "[CRITICAL]";
            default: return "[UNKNOWN]";
        }
    }

    void vlog(LogLevel level, const char* text, va_list args) {
        char* buf = new char[strlen(text)+1024];
        vsnprintf(buf, strlen(text)+1024, text, args);

        std::cout << levelString(level) << Timestamp() << " -> " << buf << std::endl;

        delete[] buf;
    }

    template <typename T>
    typename std::enable_if<is_iterable<T>::value>::type log(const T& container, LogLevel level = Debug) {
        std::cout << levelString(level) << Timestamp() << " -> ";
        for (const auto& element : container) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    void log(LogLevel level, const char* text, ...) {
        va_list args;
        va_start(args, text);
        vlog(level, text, args);
    }

    void log(const char* text, ...) {
        va_list args;
        va_start(args, text);
        vlog(Debug, text, args);
    }
}