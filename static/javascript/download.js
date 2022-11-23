// Credits: https://dev.to/shantanu_jana/save-textbox-value-to-file-using-javascript-2ljp

function downloadFile(filename, content) {
  // Downloads the poem as a file
  const element = document.createElement('a')
  const blob = new Blob([content], { type: 'plain/text' })

  // Credits a DOMString containing a URL representing the object given in the parameter.
  const fileUrl = URL.createObjectURL(blob)

  // Setting values for file url and file name
  element.setAttribute('href', fileUrl) //file location
  element.setAttribute('download', filename) // file name
  element.style.display = 'none'

  // Add element to DOM
  document.body.appendChild(element)
  element.click()

  // Removes a child node from the DOM and returns the removed node
  document.body.removeChild(element)
};
  
  window.onload = () => {
    // When download button gets clicked, downloads poem to user's computer
    document.getElementById('downloadBtn').
    addEventListener('click', e => {
        const filename = "PowerUpAsia.txt"
        const content = document.getElementById('textarea').value    
        if (filename && content) {
            downloadFile(filename, content)
        }
    });
};