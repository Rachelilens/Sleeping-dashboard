const int ledPin = 13;          
const int buzzerPin = 12;       
const int buttonPin = 11;        
const int sensorPins[] = {A0, A1, A2, A3}; 
const int valuePin = A5;        
int baselines[4] = {0, 0, 0, 0};


bool alertActive = false;       

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);


  for (int i = 0; i < 4; i++) {
    baselines[i] = analogRead(sensorPins[i]);
  }
}

void loop() {

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    if (command == "ALERT") {
      activateAlert();
    }
  }


  if (alertActive) {
    if (digitalRead(buttonPin) == LOW) {
      deactivateAlert();
    }
  } else {

    sendSensorData();
  }

  delay(1000);
}

void activateAlert() {
  alertActive = true;
  digitalWrite(ledPin, HIGH);  
  tone(buzzerPin, 1000);       
}

void deactivateAlert() {
  alertActive = false;
  digitalWrite(ledPin, LOW);   
  noTone(buzzerPin);          
}

void sendSensorData() {
  // 读取传感器数据
  int sensorValues[4];
  for (int i = 0; i < 4; i++) {
    sensorValues[i] = analogRead(sensorPins[i]) - baselines[i]; 
  }
  int value = analogRead(valuePin);


  Serial.print(value);
  Serial.print(",");
  for (int i = 0; i < 4; i++) {
    Serial.print(sensorValues[i]);
    if (i < 3) Serial.print(",");
  }
  Serial.println();
}