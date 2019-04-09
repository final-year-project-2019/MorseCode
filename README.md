# Blink to Morse Code Converter

## Running a Program

 1. Clone this on to your computer.  
 2. Go to the cloned directory and run.  
 ``` source ./bin/activate ```
 3. This activates the virtual environment necessary for execution of the program. 
 4. The program can be run as follows
 ``` python3 detect_blinks.py -p shape_predictor_68_face_landmarks.dat ```

## Working of the Program

 1. On program start, the video recording begins
 2. The user can now input morse code through blinks
 3. A  short blink count towards a dot and a long blink count towards a dash.
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/International_Morse_Code.svg/1200px-International_Morse_Code.svg.png" width="500px">
 4. The appropriate morse code characters are entered via blink.
 5. On encountering a longer period without any input, the existing morse code characters are converted to their english character equivalents
 6. 5 dots counts towards a space and 6 dots counts towards a delete.