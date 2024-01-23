import { ArrowUpOnSquareIcon, TrashIcon } from "@heroicons/react/24/outline";
import { Popover, Transition, RadioGroup } from "@headlessui/react";
import { Fragment, useState, useCallback, useEffect, useRef } from "react";
import {
  EmphasizedButton,
  EmphasizedButtonWithArrow,
  EmphasizedButtonWithBackArrow,
} from "./Buttons";
import { useDropzone } from "react-dropzone";
import fbp1 from "./TeamOnePlayerPictures/Screenshot 2024-01-22 183131.png";
import fbp2 from "./TeamOnePlayerPictures/Screenshot 2024-01-22 183250.png";
import fbp3 from "./TeamOnePlayerPictures/Screenshot 2024-01-22 183302.png";
import fbp4 from "./TeamTwoPlayerPictures/Screenshot 2024-01-22 183033.png";
import fbp5 from "./TeamTwoPlayerPictures/Screenshot 2024-01-22 183046.png";
import fbp6 from "./TeamTwoPlayerPictures/Screenshot 2024-01-22 183120.png";

function formatTime(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = Math.round(seconds % 60);

  let timeString = "";

  if (hours > 0) {
    timeString += hours + "h ";
  }

  if (minutes > 0 || hours > 0) {
    timeString += minutes + "m ";
  }

  timeString += remainingSeconds + "s";

  return timeString.trim();
}

// We have to use promises to avoid race conditions
export default function FilmUpload() {
  const [files, setFiles] = useState([]);

  const [teamNameText, setTeamNameText] = useState("");
  const inputRef = useRef(null);

  const handleTeamNameChange = (event) => {
    setTeamNameText(event.target.value);
  };

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [teamNameText]);

  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);

  let teamOnePlayerPictures = [fbp1, fbp2, fbp3];
  let teamTwoPlayerPictures = [fbp4, fbp5, fbp6];

  const [selectedTeam, setSelectedTeam] = useState(1);

  const [games, setGames] = useState([]);

  async function handleFileDrop(droppedFiles) {
    const filePromises = droppedFiles.map((file) => {
      return new Promise((resolve, reject) => {
        if (file && file.type === "video/mp4") {
          const video = document.createElement("video");
          video.preload = "metadata";

          video.onloadedmetadata = () => {
            window.URL.revokeObjectURL(video.src);
            const duration = video.duration;
            resolve({ name: file.name, duration: duration, file: file });
          };

          video.onerror = () => {
            reject(new Error(`Error loading metadata for ${file.name}`));
          };

          video.src = URL.createObjectURL(file);
        } else {
          resolve(null); // Resolve with null for non-mp4 files
        }
      });
    });

    try {
      const results = await Promise.all(filePromises);
      const validResults = results.filter((result) => result !== null); // Filter out null results
      setFiles([...files, ...validResults]);
    } catch (error) {
      console.error("Error processing files:", error);
    }
  }

  const handleNextClick = () => {
    if (currentTaskIndex < taskSet.length - 1) {
      setCurrentTaskIndex(currentTaskIndex + 1);
    }
  };

  const handleBackClick = () => {
    if (currentTaskIndex > 0) {
      setCurrentTaskIndex(currentTaskIndex - 1);
    }
  };

  const handleAddAnotherGame = () => {
    setGames([...games, { files: [...files], selectedTeam: selectedTeam }]);
    setFiles([]);
    setCurrentTaskIndex(0);
  };

  const handleSubmitJob = () => {
    if (files.length > 0) {
      setGames([...games, { files: [...files], selectedTeam: selectedTeam }]);
      setFiles([]);
    }
    console.log(games);
  };

  function FilmUploadTask() {
    const { getRootProps, getInputProps } = useDropzone({
      onDrop: handleFileDrop,
    });

    return (
      <>
        <div class="flex items-center justify-center w-full p-4 bg-white">
          <div
            className="flex flex-col items-center justify-center w-full h-42 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 hover:bg-gray-100"
            {...getRootProps()}
          >
            <ArrowUpOnSquareIcon class="w-8 h-8 my-4 text-gray-500" />
            <div class="flex flex-col items-center justify-center pt-5 pb-6">
              <p class="mb-2 text-sm text-gray-500">
                <span class="font-semibold">Click to upload</span> or drag and
                drop
              </p>
              <p class="text-xs text-gray-500">MP4</p>
            </div>
            <input {...getInputProps()} />
          </div>
        </div>
        <div className="relative grid gap-8 bg-white p-7 h-64 w-full overflow-auto">
          {files.map((item) => (
            <div
              className="-m-3 flex items-center p-2 transition duration-150 ease-in-out focus:outline-none"
              key={item.name}
            >
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-900">{item.name}</p>
                <p className="text-sm text-gray-600">
                  {formatTime(item.duration)}
                </p>
              </div>
              <div className="ml-auto">
                <a className="flex h-12 w-12 shrink-0 items-center p-1 rounded-md justify-center text-gray-600 hover:bg-gray-100">
                  <TrashIcon aria-hidden="true" className="h-7 w-7" />
                </a>
              </div>
            </div>
          ))}
        </div>
        <div class="flex items-center justify-center w-full p-2 px-4 bg-white">
          <input
            type="text"
            id="team_name"
            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg block w-full p-2.5 focus:outline-none"
            placeholder="Team Name"
            ref={inputRef}
            required
            value={teamNameText}
            onChange={handleTeamNameChange}
          />
        </div>
        <div className="flex flex-row bg-gray-50 p-4">
          <div className="flow-root rounded-md px-2 py-2 transition duration-150 ease-in-out focus:outline-none">
            <span className="flex items-center">
              <span className="text-sm font-medium text-gray-900">
                Film Upload
              </span>
            </span>
            <span className="block text-sm text-gray-600">
              Upload every clip from your game or the entire game as one clip
              and enter the name of the team to scout
            </span>
          </div>
          <div className="ml-auto grid items-center flex-grow-0">
            <EmphasizedButtonWithArrow
              text={"Next"}
              link={"#"}
              onClick={handleNextClick}
            />
          </div>
        </div>
      </>
    );
  }

  const ImageColumn = ({ images, checked }) => {
    return (
      <div
        className={
          "flex flex-col justify-center space-y-4 hover:bg-gray-100 border rounded-lg w-32 p-2.5 py-8 focus:outline-none" +
          (checked
            ? " bg-gray-200 hover:bg-gray-200 border-indigo-600"
            : " bg-gray-50  border-gray-300")
        }
      >
        {images.map((image, index) => (
          <div
            key={index}
            className="w-16 h-16 rounded-lg overflow-hidden self-center"
          >
            <img
              src={image}
              alt={`Image ${index}`}
              className="object-cover w-full h-full"
            />
          </div>
        ))}
      </div>
    );
  };

  function TeamDescriptionTask() {
    return (
      <>
        <div class="flex items-center justify-center w-full p-4 bg-white">
          <div className="flow-root rounded-md px-2 py-2 transition duration-150 ease-in-out focus:outline-none">
            <span className="text-sm font-medium text-gray-900">
              {teamNameText}
            </span>
          </div>
        </div>
        <div className="relative grid gap-8 bg-white px-7 py-4 h-80 w-full">
          <RadioGroup
            value={selectedTeam}
            onChange={setSelectedTeam}
            className="flex justify-center space-x-8"
          >
            <RadioGroup.Option value={1}>
              {({ checked }) => (
                <ImageColumn images={teamOnePlayerPictures} checked={checked} />
              )}
            </RadioGroup.Option>
            <RadioGroup.Option value={2}>
              {({ checked }) => (
                <ImageColumn images={teamTwoPlayerPictures} checked={checked} />
              )}
            </RadioGroup.Option>
          </RadioGroup>
        </div>
        <div className="flex flex-row bg-gray-50 p-4">
          <div className="flow-root rounded-md px-2 py-2 transition duration-150 ease-in-out focus:outline-none">
            <span className="flex items-center">
              <span className="text-sm font-medium text-gray-900">
                Team Description
              </span>
            </span>
            <span className="block text-sm text-gray-600">
              Select the team you want to scout
            </span>
          </div>
          <div className="ml-auto grid items-center flex-grow-0 px-2">
            <EmphasizedButtonWithBackArrow
              text={"Back"}
              link={"#"}
              onClick={handleBackClick}
            />
          </div>
          <div className=" grid items-center flex-grow-0 px-2">
            <EmphasizedButtonWithArrow
              text={"Next"}
              link={"#"}
              onClick={handleNextClick}
            />
          </div>
        </div>
      </>
    );
  }

  function ConfirmOrContinueTask() {
    return (
      <>
        <div class="flex items-center justify-center w-full p-4 bg-white">
          <div className="flow-root rounded-md px-2 py-2 transition duration-150 ease-in-out focus:outline-none">
            <span className="text-sm font-medium text-gray-900">
              Continue Adding Games Or Submit
            </span>
          </div>
        </div>
        <div className="relative grid gap-8 bg-white p-7 h-64 w-full overflow-y-auto">
          {games.map((game, index) => (
            <div
              className="-m-3 flex items-center p-2 transition duration-150 ease-in-out focus:outline-none border-b border-gray-200"
              key={index}
            >
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-900 w-48 h-6 overflow-auto text-nowrap">
                  {game.files.reduce((accumulator, file) => {
                    return accumulator + file.name + " ";
                  }, "")}
                </p>
                <p className="text-sm text-gray-600">
                  {formatTime(
                    game.files.reduce((accumulator, file) => {
                      return accumulator + file.duration;
                    }, 0)
                  )}
                </p>
              </div>
            </div>
          ))}
          <div
            className="-m-3 flex items-center p-2 transition duration-150 ease-in-out focus:outline-none border-b border-gray-200"
            key={-1}
          >
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-900 w-48 h-6 overflow-auto text-nowrap">
                {files.reduce((accumulator, file) => {
                  return accumulator + file.name + " ";
                }, "")}
              </p>
              <p className="text-sm text-gray-600">
                {formatTime(
                  files.reduce((accumulator, file) => {
                    return accumulator + file.duration;
                  }, 0)
                )}
              </p>
            </div>
          </div>
        </div>
        <div className="flex flex-row bg-gray-50 p-4">
          <div className="mx-auto grid items-center flex-grow-0 px-2">
            <EmphasizedButtonWithBackArrow
              text={"Back"}
              link={"#"}
              onClick={handleBackClick}
            />
          </div>
          <div className="mx-auto grid items-center flex-grow-0 px-2">
            <EmphasizedButtonWithArrow
              text={"Add Game"}
              link={"#"}
              onClick={handleAddAnotherGame}
            />
          </div>
          <div className="mx-auto grid items-center flex-grow-0 px-2">
            <EmphasizedButton
              text={"Submit"}
              link={"#"}
              onClick={handleSubmitJob}
            />
          </div>
        </div>
      </>
    );
  }

  let taskSet = [
    <FilmUploadTask />,
    <TeamDescriptionTask />,
    <ConfirmOrContinueTask />,
  ];

  return (
    <Popover>
      {({ open }) => (
        <>
          <Popover.Button className="focus:outline-none">
            <EmphasizedButton text={"Upload Film"} link={"#"} />
          </Popover.Button>
          <Transition
            as={Fragment}
            enter="transition ease-out duration-200"
            enterFrom="opacity-0 translate-y-1"
            enterTo="opacity-100 translate-y-0"
            leave="transition ease-in duration-150"
            leaveFrom="opacity-100 translate-y-0"
            leaveTo="opacity-0 translate-y-1"
          >
            <Popover.Panel className="absolute left-1/2 z-10 mt-3 w-screen max-w-sm -translate-x-1/2 transform px-4 sm:px-0 lg:max-w-3xl">
              <div className="overflow-hidden rounded-lg shadow-lg ring-1 ring-black/5">
                {<div key={currentTaskIndex}>{taskSet[currentTaskIndex]}</div>}
              </div>
            </Popover.Panel>
          </Transition>
        </>
      )}
    </Popover>
  );
}
