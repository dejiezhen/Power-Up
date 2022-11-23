let recognizing
const clearBtn = document.querySelector("#clearbtn")
window.SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;

const recognition = new SpeechRecognition()

const reset = () => {
  // Resetting initial parameters
  recognizing = false;
  document.querySelector('#mic-status').innerHTML = 'Click to Speak'
}

recognition.continuous = true;
reset();
recognition.onend = reset();

recognition.onresult = (e) => {
  // Appends speech recognition data to text area after finishing
  for (var i = e.resultIndex; i < e.results.length; ++i) {
      if (e.results[i].isFinal) {
        textarea.value += e.results[i][0].transcript;
      }
  }
}

const stopRecognition = () => {
  // Stops speech recognition
  recognition.stop();
  reset();
}

const startRecognition = () => {
  // Start speech recognition
  poem_text.value = ''
  recognition.start();
  recognizing = true;
  document.querySelector('#mic-status').innerHTML = 'Click to Stop'
}

const toggleSpeechRecognition = () => {
  // Toggles speech recognition on and off
  recognizing 
    ? stopRecognition()
    : startRecognition()
}

clearBtn.onclick = (e) => {
  // Clear text field when button is clicked
  e.preventDefault()
  poem_text.value = ''
}
