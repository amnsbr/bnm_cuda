#ifndef HELPERS
#define HELPERS

float get_env_or_default(std::string key, double value_default = 0.0) {
    const char* value = std::getenv(key.c_str());
    if (value != nullptr) {
        return atof(value);
    } else {
        return value_default;
    }
}

int get_env_or_default(std::string key, int value_default = 0) {
    const char* value = std::getenv(key.c_str());
    if (value != nullptr) {
        return atoi(value);
    } else {
        return value_default;
    }
}

#endif