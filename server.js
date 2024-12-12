import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';


const app = express();

app.use(cors());
app.use(express.json());

const DB_URI = 'mongodb+srv://ilens:q1w2e3@sleeping.trfss.mongodb.net/mongodbVSCodePlaygroundDB?retryWrites=true&w=majority';

mongoose.connect(DB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log('Connected to MongoDB Cloud'))
  .catch((error) => console.error('Error connecting to MongoDB:', error));


const sleepSchema = new mongoose.Schema({
  Timestamp: { type: String, required: true },
  Value: { type: Number, required: true },
  FSR0: { type: Number },
  FSR1: { type: Number },
  FSR2: { type: Number },
  FSR3: { type: Number },
  "Sleeping posture": { type: String, required: true },
  "Accuracy value": { type: Number },
  Score1: { type: Number },
  Score2: { type: Number },
  Score3: { type: Number },
  Score4: { type: Number },
  Score5: { type: Number },
  Score6: { type: Number },
});


const SleepData = mongoose.model('final', sleepSchema, 'final');

app.get('/api/sleep-data', async (req, res) => {
  try {
    const { date } = req.query;
    if (!date) {
      return res.status(400).send('Date parameter is required');
    }

    const allowedDates = [
      '2024-12-03',
      '2024-12-04',
      '2024-12-05',
      '2024-12-06',
      '2024-12-07',
      '2024-12-08',
      '2024-12-09',
      '2024-12-11',
    ];

    if (!allowedDates.includes(date)) {
      return res.status(400).send('Invalid date parameter');
    }

    const data = await SleepData.find({ Timestamp: { $regex: `^${date}` } });
    console.log(`Data for ${date}:`, data);
    res.json(data);
  } catch (error) {
    console.error('Error:', error);
    res.status(500).send(error.message);
  }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));