// server.js
const express = require("express")
const multer = require("multer")
const cors = require("cors")
const path = require("path")
const tf = require("@tensorflow/tfjs-node") // For running your ML model

const app = express()
app.use(cors())

// Configure multer for handling file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/")
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname))
  },
})

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/")) {
      cb(null, true)
    } else {
      cb(new Error("Not an image! Please upload an image."), false)
    }
  },
})

// Load your model (adjust path as needed)
let model
async function loadModel() {
  try {
    // Replace this with your model loading logic
    model = await tf.loadLayersModel("file://./model/model.json")
    console.log("Model loaded successfully")
  } catch (error) {
    console.error("Error loading model:", error)
  }
}

loadModel()

// Endpoint for helmet detection
app.post("/api/detect-helmet", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file provided" })
    }

    // Read the uploaded image
    const imagePath = req.file.path

    // Load and preprocess the image for your model
    const image = await tf.node.decodeImage(
      require("fs").readFileSync(imagePath)
    )

    // Preprocess image according to your model's requirements
    const processedImage = preprocessImage(image)

    // Make prediction
    const prediction = await model.predict(processedImage)

    // Process the prediction results
    const result = {
      wearing_helmet: prediction[0] > 0.5, // Adjust threshold as needed
      confidence: prediction[0].dataSync()[0],
      // Add bounding box coordinates if your model provides them
      bounding_box: [100, 100, 200, 200], // Example values
    }

    // Clean up: delete uploaded file
    require("fs").unlinkSync(imagePath)

    res.json(result)
  } catch (error) {
    console.error("Error processing image:", error)
    res.status(500).json({ error: "Error processing image" })
  }
})

// Helper function to preprocess image
function preprocessImage(image) {
  // Resize image to match your model's input size
  const resized = tf.image.resizeBilinear(image, [224, 224]) // Adjust size as needed

  // Normalize pixel values
  const normalized = resized.div(255.0)

  // Add batch dimension
  return normalized.expandDims(0)
}

const PORT = process.env.PORT || 5000
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})
