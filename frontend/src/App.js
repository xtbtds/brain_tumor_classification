import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionField, setPredictionField] = useState({})

  const handleSelectImage = (e) => {
    // ___________________Ensure that only images are selected_________________
    const file = e.target.files[0];
    if (file.type.split('/')[0] === 'image') {
      setSelectedImage(file);
    } else {
        console.log("Only images are accepted.")
    }
  }

  const handleUploadImage = async () => {
    if (!selectedImage) {
      alert("Please select an image and then click 'Upload image'.")
      return;
    }
    const formData = new FormData();
    formData.append('file', selectedImage);
    try {
      const res = await fetch('http://0.0.0.0:8080/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setPredictionField(predictionField => ({
        ...predictionField,
        ...data.result
      }));
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <div className="App">

        <div className="buttons">
            <button className="uploadButton" onClick={() => document.getElementById('image-input').click()}>Select Image</button>
            <input type="file" id="image-input" style={{ display: 'none' }} onChange={handleSelectImage} />
        </div>

        <div className="buttons">
            <button className="uploadButton" onClick={handleUploadImage}>Upload Image</button>
        </div>

        <div className="predictions">
            <h3 className="startPred">Your Predictions:</h3>
            <p>Glioma tumor: {predictionField.glioma_tumor}%</p>
            <p>No tumor: {predictionField.no_tumor}%</p>
            <p>Meningioma tumor: {predictionField.meningioma_tumor}%</p>
            <p>Pituitary tumor: {predictionField.pituitary_tumor}%</p>
        </div>
    </div>
  );
}

export default App;