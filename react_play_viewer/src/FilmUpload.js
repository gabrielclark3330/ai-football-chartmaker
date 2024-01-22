import { ArrowUpOnSquareIcon, TrashIcon } from "@heroicons/react/24/outline";
import { Popover, Transition } from "@headlessui/react";
import { Fragment, useState, useCallback, useEffect } from "react";
import { EmphasizedButton, EmphasizedButtonWithArrow } from "./Buttons";
import { useDropzone } from "react-dropzone";

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

  async function handleFileDrop(droppedFiles) {
    const filePromises = droppedFiles.map((file) => {
      return new Promise((resolve, reject) => {
        if (file && file.type === "video/mp4") {
          const video = document.createElement("video");
          video.preload = "metadata";

          video.onloadedmetadata = () => {
            window.URL.revokeObjectURL(video.src);
            const duration = video.duration;
            console.log(duration);
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

  const { getRootProps, getInputProps } = useDropzone({
    onDrop: handleFileDrop,
  });

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
                <div class="flex items-center justify-center w-full p-4 bg-white">
                  <div
                    className="flex flex-col items-center justify-center w-full h-42 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 hover:bg-gray-100"
                    {...getRootProps()}
                  >
                    <ArrowUpOnSquareIcon class="w-8 h-8 my-4 text-gray-500" />
                    <div class="flex flex-col items-center justify-center pt-5 pb-6">
                      <p class="mb-2 text-sm text-gray-500">
                        <span class="font-semibold">Click to upload</span> or
                        drag and drop
                      </p>
                      <p class="text-xs text-gray-500">MP4</p>
                    </div>
                    <input
                      id="dropzone-file"
                      type="file"
                      class="hidden"
                      {...getInputProps()}
                    />
                  </div>
                </div>
                <div className="relative grid gap-8 bg-white p-7 h-64 w-full overflow-auto">
                  {files.map((item) => (
                    <div
                      className="-m-3 flex items-center p-2 transition duration-150 ease-in-out focus:outline-none"
                      key={item.name}
                    >
                      <div className="ml-4">
                        <p className="text-sm font-medium text-gray-900">
                          {item.name}
                        </p>
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
                <div className="flex flex-row bg-gray-50 p-4">
                  <a
                    href="##"
                    className="flow-root rounded-md px-2 py-2 transition duration-150 ease-in-out focus:outline-none"
                  >
                    <span className="flex items-center">
                      <span className="text-sm font-medium text-gray-900">
                        Documentation
                      </span>
                    </span>
                    <span className="block text-sm text-gray-600">
                      Start integrating products and tools
                    </span>
                  </a>
                  <div className="ml-auto grid items-center flex-grow-0">
                      <EmphasizedButtonWithArrow text={"Next"} />
                  </div>
                </div>
              </div>
            </Popover.Panel>
          </Transition>
        </>
      )}
    </Popover>
  );
}
