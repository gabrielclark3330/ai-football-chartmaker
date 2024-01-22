export default function Footer() {
  return (
    <div className="p-4 mt-auto">
      <footer>
        <div class="w-full mx-auto max-w-screen-xl p-4 md:flex md:items-center md:justify-between">
          <span class="text-sm text-gray-600 sm:text-center">
            © 2023{" "}
              2%Football. All Rights Reserved.
          </span>
          <ul class="flex flex-wrap items-center mt-3 text-sm font-medium text-gray-600 sm:mt-0">
            <li>
              <a href="#" class="hover:underline me-4 md:me-6">
                YouTube
              </a>
            </li>
            <li>
              <a href="#" class="hover:underline me-4 md:me-6">
                Instagram
              </a>
            </li>
            <li>
              <a href="#" class="hover:underline me-4 md:me-6">
                X
              </a>
            </li>
            <li>
              <a href="#" class="hover:underline">
                Contact
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </div>
  );
}
