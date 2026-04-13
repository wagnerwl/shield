#include <WiFi.h>
#include <ArduinoHA.h>

const char* WIFI_SSID     = "DeinWLAN";
const char* WIFI_PASSWORD = "DeinWLANPasswort";
const char* MQTT_SERVER   = "192.168.1.XXX";
const char* MQTT_USER     = "portenta";
const char* MQTT_PASSWORD = "geheimespasswort";

// Eindeutige Geräte-ID (z.B. MAC-Adresse als Byte-Array)
byte deviceId[] = {0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56};

WiFiClient wifiClient;
HADevice device(deviceId, sizeof(deviceId));
HAMqtt mqtt(wifiClient, device);

// Sensor definieren
HASensorNumber tempSensor("temperatur", HASensorNumber::PrecisionP1);

void setup() {
  Serial.begin(115200);

  // Gerätename für HA
  device.setName("Portenta H7 Labor");
  device.setModel("Arduino Portenta H7");
  device.setManufacturer("Arduino");

  // Sensor konfigurieren
  tempSensor.setName("Temperatur");
  tempSensor.setUnitOfMeasurement("°C");
  tempSensor.setDeviceClass("temperature");

  // WLAN verbinden
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWLAN verbunden!");

  // MQTT verbinden
  mqtt.begin(MQTT_SERVER, MQTT_USER, MQTT_PASSWORD);
}

void loop() {
  mqtt.loop();

  // Alle 10 Sekunden Wert senden
  static unsigned long lastUpdate = 0;
  if (millis() - lastUpdate >= 10000) {
    lastUpdate = millis();
    float temp = 22.5;  // Hier echten Sensor einlesen
    tempSensor.setValue(temp);
    Serial.println("Temperatur gesendet: " + String(temp));
  }
}