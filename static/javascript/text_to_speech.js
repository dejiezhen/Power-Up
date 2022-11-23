const utterThis = new SpeechSynthesisUtterance()
const speaker = document.querySelector(".speaker-button-large")
const poem_text = document.querySelector("#textarea")
let poem = ""
let voices = []

'speechSynthesis' in window 
    ? console.log("Web Speech API supported!") 
    : console.log("Web Speech API not supported")

const synth = window.speechSynthesis

speaker.onclick = (e) => {
    // Text to speech when the speaker gets clicked
    e.preventDefault()
    voices = window.speechSynthesis.getVoices();
    poem = poem_text.value
    utterThis.voice = voices[39]
    utterThis.text = poem
    synth.speak(utterThis)
}
